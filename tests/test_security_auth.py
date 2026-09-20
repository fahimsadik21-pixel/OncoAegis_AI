from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timedelta, timezone
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
from jose.utils import base64url_encode

from app.api.auth_api import (
    login,
    logout,
    me,
    oauth_callback_get,
    providers,
    refresh,
    register,
)
from app.schemas.auth import LoginRequest, LogoutRequest, RefreshRequest, RegisterRequest
from app.security.auth import decode_access_token
from app.security.dependencies import get_current_user
from app.security.password import hash_password, verify_password
from app.security.permissions import Permission, has_permission
from app.security.oauth import (
    OAuthProvider,
    OAuthProviderConfig,
    OAuthIdentity,
    _verify_identity_token,
    exchange_authorization_code,
)
from app.security.roles import UserRole
from app.security.store import AuthStore, DuplicateUserError, set_auth_store


class SecurityAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = AuthStore(Path(self.directory.name) / "auth.sqlite3")
        set_auth_store(self.store)

    def tearDown(self) -> None:
        set_auth_store(None)
        self.directory.cleanup()

    @staticmethod
    def _credentials(token: str) -> HTTPAuthorizationCredentials:
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    def test_password_hash_is_salted_and_verifies(self):
        first = hash_password("correct horse battery staple")
        second = hash_password("correct horse battery staple")
        self.assertNotEqual(first, second)
        self.assertNotIn("correct horse", first)
        self.assertTrue(verify_password("correct horse battery staple", first))
        self.assertFalse(verify_password("wrong password", first))

    def test_register_login_me_refresh_and_logout(self):
        created = register(
            RegisterRequest(
                email="Doctor@Example.com",
                username="doctor-one",
                password="correct horse battery staple",
            )
        )
        self.assertEqual(created.user.email, "doctor@example.com")
        self.assertEqual(created.user.role, UserRole.VIEWER.value)
        self.assertTrue(created.access_token)
        self.assertTrue(created.refresh_token)

        access_credentials = self._credentials(created.access_token)
        current = get_current_user(access_credentials)
        self.assertEqual(current.user_id, created.user.user_id)
        self.assertEqual(me(current).email, "doctor@example.com")

        signed_in = login(
            LoginRequest(
                email="doctor@example.com",
                password="correct horse battery staple",
            )
        )
        self.assertNotEqual(signed_in.access_token, created.access_token)

        rotated = refresh(RefreshRequest(refresh_token=signed_in.refresh_token))
        with self.assertRaises(HTTPException) as old_refresh:
            refresh(RefreshRequest(refresh_token=signed_in.refresh_token))
        self.assertEqual(old_refresh.exception.status_code, 401)

        rotated_credentials = self._credentials(rotated.access_token)
        rotated_current = get_current_user(rotated_credentials)
        logout(
            LogoutRequest(refresh_token=rotated.refresh_token),
            rotated_credentials,
            rotated_current,
        )
        with self.assertRaises(HTTPException) as revoked:
            get_current_user(rotated_credentials)
        self.assertEqual(revoked.exception.status_code, 401)

    def test_duplicate_email_and_invalid_login_are_safe(self):
        request = RegisterRequest(
            email="viewer@example.com",
            password="correct horse battery staple",
        )
        register(request)
        with self.assertRaises(HTTPException) as duplicate:
            register(request)
        self.assertEqual(duplicate.exception.status_code, 409)
        with self.assertRaises(HTTPException) as invalid:
            login(LoginRequest(email="viewer@example.com", password="incorrect"))
        self.assertEqual(invalid.exception.status_code, 401)

    def test_provider_registry_and_oauth_state_are_ready_without_fake_login(self):
        status_items = {item.provider: item for item in providers()}
        self.assertTrue(status_items["local"].enabled)
        self.assertFalse(status_items["google"].enabled)
        self.assertFalse(status_items["apple"].enabled)

        state = self.store.create_oauth_state("google")
        self.assertTrue(self.store.oauth_state_is_valid(state=state, provider="google"))
        self.assertTrue(self.store.consume_oauth_state(state=state, provider="google"))
        self.assertFalse(self.store.oauth_state_is_valid(state=state, provider="google"))

        oauth_user = self.store.create_user(
            email="oauth@example.com",
            password_hash=None,
            provider="google",
            email_verified=True,
        )
        self.store.link_oauth_identity(
            provider="google", provider_subject="google-subject", user_id=oauth_user.user_id
        )
        self.assertEqual(
            self.store.get_user_by_oauth(
                provider="google", provider_subject="google-subject"
            ).user_id,
            oauth_user.user_id,
        )

    def test_permissions_use_one_role_source(self):
        user = register(
            RegisterRequest(
                email="researcher@example.com",
                password="correct horse battery staple",
            )
        )
        context = get_current_user(self._credentials(user.access_token))
        self.assertTrue(has_permission(context, Permission.VIEW_RESULT))
        self.assertFalse(has_permission(context, Permission.MANAGE_USERS))
        self.assertEqual(decode_access_token(user.access_token)["type"], "access")

    def test_provider_code_exchange_verifies_signed_identity_token(self):
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        private_numbers = private_key.private_numbers().public_numbers
        jwks = {
            "keys": [
                {
                    "kty": "RSA",
                    "kid": "test-key",
                    "use": "sig",
                    "alg": "RS256",
                    "n": base64url_encode(
                        private_numbers.n.to_bytes((private_numbers.n.bit_length() + 7) // 8, "big")
                    ).decode("ascii"),
                    "e": base64url_encode(
                        private_numbers.e.to_bytes((private_numbers.e.bit_length() + 7) // 8, "big")
                    ).decode("ascii"),
                }
            ]
        }
        private_pem = private_key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        )
        config = OAuthProviderConfig(
            provider=OAuthProvider.GOOGLE,
            client_id="google-client",
            client_secret="google-secret",
            redirect_uri="https://app.example.test/auth/google/callback",
            authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
            token_endpoint="https://oauth2.googleapis.com/token",
            jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
            issuer="https://accounts.google.com",
            scopes=("openid", "email", "profile"),
        )
        now = datetime.now(timezone.utc)
        identity_token = jwt.encode(
            {
                "iss": "https://accounts.google.com",
                "aud": "google-client",
                "sub": "provider-subject",
                "email": "provider@example.com",
                "email_verified": True,
                "name": "Provider User",
                "iat": now,
                "exp": now + timedelta(minutes=5),
            },
            private_pem,
            algorithm="RS256",
            headers={"kid": "test-key"},
        )
        with patch(
            "app.security.oauth._request_json",
            return_value={"id_token": identity_token},
        ) as request_json, patch(
            "app.security.oauth._fetch_jwks", return_value=jwks
        ):
            identity = exchange_authorization_code(config, code="one-time-code")
        self.assertEqual(identity.provider, OAuthProvider.GOOGLE)
        self.assertEqual(identity.email, "provider@example.com")
        request_json.assert_called_once()

    def test_verified_oauth_callback_links_user_and_issues_session(self):
        state = self.store.create_oauth_state("google")
        env = {
            "ONCOAEGIS_GOOGLE_CLIENT_ID": "google-client",
            "ONCOAEGIS_GOOGLE_CLIENT_SECRET": "google-secret",
            "ONCOAEGIS_GOOGLE_REDIRECT_URI": "https://app.example.test/auth/google/callback",
        }
        identity = OAuthIdentity(
            provider=OAuthProvider.GOOGLE,
            subject="verified-subject",
            email="linked@example.com",
            display_name="Linked User",
            email_verified=True,
        )
        with patch.dict(os.environ, env, clear=False), patch(
            "app.api.auth_api.exchange_authorization_code", return_value=identity
        ):
            response = oauth_callback_get(
                "google", code="one-time-code", state=state
            )
        self.assertEqual(response.user.email, "linked@example.com")
        self.assertIsNotNone(
            self.store.get_user_by_oauth(
                provider="google", provider_subject="verified-subject"
            )
        )


if __name__ == "__main__":
    unittest.main()
