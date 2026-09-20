"""Google and Apple OAuth code exchange and signed identity verification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from jose import JWTError, jwt


class OAuthProvider(str, Enum):
    GOOGLE = "google"
    APPLE = "apple"


class OAuthExchangeError(RuntimeError):
    """Provider exchange or identity verification failed safely."""


@dataclass(frozen=True)
class OAuthProviderConfig:
    provider: OAuthProvider
    client_id: str | None
    client_secret: str | None
    redirect_uri: str | None
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str
    issuer: str
    scopes: tuple[str, ...]
    team_id: str | None = None
    key_id: str | None = None
    private_key: str | None = None

    def generated_client_secret(self) -> str | None:
        if self.client_secret:
            return self.client_secret
        if self.provider is not OAuthProvider.APPLE:
            return None
        if not self.client_id or not self.team_id or not self.key_id or not self.private_key:
            return None
        try:
            now = int(time.time())
            return jwt.encode(
                {
                    "iss": self.team_id,
                    "iat": now,
                    "exp": now + 86400 * 180,
                    "aud": "https://appleid.apple.com",
                    "sub": self.client_id,
                },
                self.private_key,
                algorithm="ES256",
                headers={"kid": self.key_id},
            )
        except (TypeError, ValueError, JWTError) as exc:
            raise OAuthExchangeError("Apple client secret could not be generated") from exc

    @property
    def enabled(self) -> bool:
        if not self.client_id or not self.redirect_uri:
            return False
        try:
            return bool(self.generated_client_secret())
        except OAuthExchangeError:
            return False


@dataclass(frozen=True)
class OAuthIdentity:
    """Identity normalized only after a provider-signed token is verified."""

    provider: OAuthProvider
    subject: str
    email: str
    display_name: str | None = None
    email_verified: bool = False


_jwks_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_jwks_lock = threading.Lock()


def _read_private_key() -> str | None:
    raw = os.getenv("ONCOAEGIS_APPLE_PRIVATE_KEY")
    path_value = os.getenv("ONCOAEGIS_APPLE_PRIVATE_KEY_PATH")
    if not raw and path_value:
        try:
            raw = Path(path_value).read_text(encoding="utf-8")
        except OSError as exc:
            raise OAuthExchangeError("Apple private key could not be read") from exc
    return raw.replace("\\n", "\n") if raw else None


def get_provider_config(provider: OAuthProvider | str) -> OAuthProviderConfig:
    value = provider.value if isinstance(provider, OAuthProvider) else str(provider).lower()
    if value == OAuthProvider.GOOGLE.value:
        return OAuthProviderConfig(
            provider=OAuthProvider.GOOGLE,
            client_id=os.getenv("ONCOAEGIS_GOOGLE_CLIENT_ID"),
            client_secret=os.getenv("ONCOAEGIS_GOOGLE_CLIENT_SECRET"),
            redirect_uri=os.getenv("ONCOAEGIS_GOOGLE_REDIRECT_URI"),
            authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
            token_endpoint="https://oauth2.googleapis.com/token",
            jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
            issuer="https://accounts.google.com",
            scopes=("openid", "email", "profile"),
        )
    if value == OAuthProvider.APPLE.value:
        return OAuthProviderConfig(
            provider=OAuthProvider.APPLE,
            client_id=os.getenv("ONCOAEGIS_APPLE_CLIENT_ID"),
            client_secret=os.getenv("ONCOAEGIS_APPLE_CLIENT_SECRET"),
            redirect_uri=os.getenv("ONCOAEGIS_APPLE_REDIRECT_URI"),
            authorization_endpoint="https://appleid.apple.com/auth/authorize",
            token_endpoint="https://appleid.apple.com/auth/token",
            jwks_uri="https://appleid.apple.com/auth/keys",
            issuer="https://appleid.apple.com",
            scopes=("name", "email"),
            team_id=os.getenv("ONCOAEGIS_APPLE_TEAM_ID"),
            key_id=os.getenv("ONCOAEGIS_APPLE_KEY_ID"),
            private_key=_read_private_key(),
        )
    raise ValueError("Unsupported OAuth provider")


def provider_status() -> list[dict[str, object]]:
    return [
        {
            "provider": "local",
            "label": "Email and password",
            "enabled": True,
            "status": "ready",
        },
        *[
            {
                "provider": config.provider.value,
                "label": config.provider.value.title(),
                "enabled": config.enabled,
                "status": "ready" if config.enabled else "not_configured",
            }
            for config in (
                get_provider_config(OAuthProvider.GOOGLE),
                get_provider_config(OAuthProvider.APPLE),
            )
        ],
    ]


def build_authorization_url(config: OAuthProviderConfig, *, state: str) -> str:
    if not config.enabled or not config.client_id or not config.redirect_uri:
        raise RuntimeError(f"{config.provider.value} OAuth is not configured")
    values = {
        "client_id": config.client_id,
        "redirect_uri": config.redirect_uri,
        "response_type": "code",
        "scope": " ".join(config.scopes),
        "state": state,
    }
    if config.provider is OAuthProvider.APPLE:
        values["response_mode"] = "form_post"
    else:
        values["access_type"] = "offline"
        values["prompt"] = "select_account"
    return f"{config.authorization_endpoint}?{urlencode(values)}"


def _request_json(
    url: str,
    *,
    form: dict[str, str] | None = None,
    authorization: str | None = None,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if form is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        data = urlencode(form).encode("utf-8")
    if authorization:
        headers["Authorization"] = authorization
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(request, timeout=10) as response:
            body = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise OAuthExchangeError("OAuth provider request failed") from exc
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OAuthExchangeError("OAuth provider returned an invalid response") from exc
    if not isinstance(parsed, dict):
        raise OAuthExchangeError("OAuth provider returned an invalid response")
    return parsed


def _fetch_jwks(config: OAuthProviderConfig) -> dict[str, Any]:
    now = time.time()
    with _jwks_lock:
        cached = _jwks_cache.get(config.jwks_uri)
        if cached and cached[0] > now:
            return cached[1]
    keys = _request_json(config.jwks_uri)
    if not isinstance(keys.get("keys"), list):
        raise OAuthExchangeError("OAuth provider returned invalid signing keys")
    with _jwks_lock:
        _jwks_cache[config.jwks_uri] = (now + 3600, keys)
    return keys


def _verify_identity_token(
    config: OAuthProviderConfig, identity_token: str
) -> OAuthIdentity:
    try:
        header = jwt.get_unverified_header(identity_token)
    except JWTError as exc:
        raise OAuthExchangeError("OAuth identity token header is invalid") from exc
    algorithm = header.get("alg")
    key_id = header.get("kid")
    if algorithm not in {"RS256", "ES256"} or not key_id:
        raise OAuthExchangeError("OAuth identity token uses an unsupported key")
    jwks = _fetch_jwks(config)
    key = next(
        (candidate for candidate in jwks["keys"] if candidate.get("kid") == key_id),
        None,
    )
    if key is None:
        raise OAuthExchangeError("OAuth identity token key was not found")
    try:
        decode_options = {}
        if config.provider is OAuthProvider.GOOGLE:
            decode_options = {"verify_iss": False}
        claims = jwt.decode(
            identity_token,
            key,
            algorithms=[algorithm],
            audience=config.client_id,
            issuer=config.issuer if config.provider is OAuthProvider.APPLE else None,
            options=decode_options,
        )
    except (JWTError, TypeError, ValueError) as exc:
        raise OAuthExchangeError("OAuth identity token could not be verified") from exc
    if config.provider is OAuthProvider.GOOGLE and claims.get("iss") not in {
        "https://accounts.google.com",
        "accounts.google.com",
    }:
        raise OAuthExchangeError("OAuth identity token issuer is invalid")
    subject = claims.get("sub")
    email = claims.get("email")
    if not isinstance(subject, str) or not subject or not isinstance(email, str) or not email:
        raise OAuthExchangeError("OAuth identity did not contain a usable account")
    verified = claims.get("email_verified")
    email_verified = verified is True or str(verified).lower() == "true"
    if not email_verified:
        raise OAuthExchangeError("OAuth email address is not verified")
    return OAuthIdentity(
        provider=config.provider,
        subject=subject,
        email=email,
        display_name=claims.get("name") if isinstance(claims.get("name"), str) else None,
        email_verified=True,
    )


def exchange_authorization_code(
    config: OAuthProviderConfig, *, code: str
) -> OAuthIdentity:
    if not config.enabled or not config.client_id or not config.redirect_uri:
        raise OAuthExchangeError(f"{config.provider.value.title()} OAuth is not configured")
    if not code.strip():
        raise OAuthExchangeError("OAuth authorization code is missing")
    client_secret = config.generated_client_secret()
    if not client_secret:
        raise OAuthExchangeError("OAuth client secret is not configured")
    tokens = _request_json(
        config.token_endpoint,
        form={
            "client_id": config.client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": config.redirect_uri,
        },
    )
    identity_token = tokens.get("id_token")
    if not isinstance(identity_token, str) or not identity_token:
        raise OAuthExchangeError("OAuth provider did not return an identity token")
    return _verify_identity_token(config, identity_token)
