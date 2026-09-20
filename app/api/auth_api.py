"""Authentication API: local credentials plus OAuth provider hand-off points."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    OAuthStartResponse,
    ProviderStatus,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.security.auth import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
)
from app.security.dependencies import get_current_user
from app.security.oauth import (
    OAuthProvider,
    OAuthExchangeError,
    build_authorization_url,
    exchange_authorization_code,
    get_provider_config,
    provider_status,
)
from app.security.password import hash_password
from app.security.security_config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from app.security.store import (
    AuthStoreError,
    DuplicateUserError,
    get_auth_store,
    normalize_email,
)


router = APIRouter(prefix="/auth", tags=["authentication"])
_bearer = HTTPBearer(auto_error=False)


def _user_response(user) -> UserResponse:
    return UserResponse.model_validate(user.to_public_dict())


def _token_expiry(payload: dict) -> datetime:
    return datetime.fromtimestamp(float(payload["exp"]), tz=timezone.utc)


def _issue_token_response(user) -> TokenResponse:
    store = get_auth_store()
    access_token = create_access_token(
        user.user_id,
        user.role,
        username=user.username,
        email=user.email,
    )
    refresh_token = create_refresh_token(
        user.user_id,
        user.role,
        username=user.username,
        email=user.email,
    )
    access_payload = decode_access_token(access_token)
    refresh_payload = decode_refresh_token(refresh_token)
    if not access_payload or not refresh_payload:
        raise HTTPException(status_code=503, detail="Unable to create a secure session")

    try:
        store.create_session(
            jti=str(access_payload["jti"]),
            user_id=user.user_id,
            token_type="access",
            expires_at=_token_expiry(access_payload),
        )
        store.create_session(
            jti=str(refresh_payload["jti"]),
            user_id=user.user_id,
            token_type="refresh",
            expires_at=_token_expiry(refresh_payload),
        )
    except AuthStoreError as exc:
        raise HTTPException(status_code=503, detail="Unable to persist secure session") from exc

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=_user_response(user),
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest) -> TokenResponse:
    try:
        user = get_auth_store().create_user(
            email=normalize_email(request.email),
            password_hash=hash_password(request.password),
            username=request.username,
        )
    except DuplicateUserError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _issue_token_response(user)


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest) -> TokenResponse:
    from app.security.password import verify_password

    try:
        user = get_auth_store().get_user_by_email(request.email)
    except ValueError:
        user = None
    if user is None or not user.password_hash or not verify_password(
        request.password, user.password_hash
    ):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    return _issue_token_response(user)


@router.post("/refresh", response_model=TokenResponse)
def refresh(request: RefreshRequest) -> TokenResponse:
    payload = decode_refresh_token(request.refresh_token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    store = get_auth_store()
    user_id = str(payload["sub"])
    if not store.is_session_active(
        jti=str(payload["jti"]), user_id=user_id, token_type="refresh"
    ):
        raise HTTPException(status_code=401, detail="Refresh session is revoked or expired")
    user = store.get_user_by_id(user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User session is not active")

    # Rotate the refresh token so a replayed refresh token cannot create a new
    # session after the first successful use.
    store.revoke_session(str(payload["jti"]))
    return _issue_token_response(user)


@router.post("/logout")
def logout(
    request: LogoutRequest | None = None,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    current_user=Depends(get_current_user),
) -> dict[str, str]:
    store = get_auth_store()
    if credentials is not None:
        payload = decode_access_token(credentials.credentials)
        if payload and str(payload.get("sub")) == current_user.user_id:
            store.revoke_session(str(payload["jti"]))
    if request and request.refresh_token:
        payload = decode_refresh_token(request.refresh_token)
        if payload and str(payload.get("sub")) == current_user.user_id:
            store.revoke_session(str(payload["jti"]))
    return {"status": "signed_out"}


@router.get("/me", response_model=UserResponse)
def me(current_user=Depends(get_current_user)) -> UserResponse:
    user = get_auth_store().get_user_by_id(current_user.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User session is not active")
    return _user_response(user)


@router.get("/providers", response_model=list[ProviderStatus])
def providers() -> list[ProviderStatus]:
    return [ProviderStatus.model_validate(item) for item in provider_status()]


@router.get("/oauth/{provider}/start", response_model=OAuthStartResponse)
def oauth_start(provider: str) -> OAuthStartResponse:
    try:
        oauth_provider = OAuthProvider(provider.lower())
        config = get_provider_config(oauth_provider)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Unsupported OAuth provider") from exc
    if not config.enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{oauth_provider.value.title()} sign-in is not configured. Set its "
                "client ID, client secret, and redirect URI first."
            ),
        )
    state = get_auth_store().create_oauth_state(oauth_provider.value)
    return OAuthStartResponse(
        provider=oauth_provider.value,
        authorization_url=build_authorization_url(config, state=state),
        state_expires_in=600,
    )


def _oauth_provider(provider: str) -> tuple[OAuthProvider, object]:
    try:
        oauth_provider = OAuthProvider(provider.lower())
        return oauth_provider, get_provider_config(oauth_provider)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Unsupported OAuth provider") from exc


def _complete_oauth(provider: str, *, code: str, state: str) -> TokenResponse:
    oauth_provider, config = _oauth_provider(provider)
    store = get_auth_store()
    if not store.consume_oauth_state(state=state, provider=oauth_provider.value):
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")
    try:
        identity = exchange_authorization_code(config, code=code)
        user = store.get_user_by_oauth(
            provider=identity.provider.value,
            provider_subject=identity.subject,
        )
        if user is None:
            user = store.get_user_by_email(normalize_email(identity.email))
            if user is None:
                user = store.create_user(
                    email=normalize_email(identity.email),
                    password_hash=None,
                    provider=identity.provider.value,
                    email_verified=identity.email_verified,
                )
            elif not user.is_active:
                raise HTTPException(status_code=403, detail="Account is disabled")
            store.link_oauth_identity(
                provider=identity.provider.value,
                provider_subject=identity.subject,
                user_id=user.user_id,
            )
    except HTTPException:
        raise
    except DuplicateUserError as exc:
        raise HTTPException(status_code=409, detail="OAuth identity is already linked") from exc
    except (OAuthExchangeError, ValueError, AuthStoreError) as exc:
        raise HTTPException(status_code=502, detail="OAuth sign-in could not be completed") from exc
    return _issue_token_response(user)


@router.get("/oauth/{provider}/callback", response_model=TokenResponse)
def oauth_callback_get(
    provider: str,
    code: str = Query(..., min_length=1),
    state: str = Query(..., min_length=1),
) -> TokenResponse:
    return _complete_oauth(provider, code=code, state=state)


@router.post("/oauth/{provider}/callback", response_model=TokenResponse)
def oauth_callback_form(
    provider: str,
    code: str = Form(..., min_length=1),
    state: str = Form(..., min_length=1),
) -> TokenResponse:
    """Accept Apple's form_post callback as well as Google's query callback."""

    return _complete_oauth(provider, code=code, state=state)
