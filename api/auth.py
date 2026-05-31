"""
src/api/auth.py
────────────────
JWT-based authentication for the FastAPI layer.

  - Token creation  : create_access_token(data)
  - Token decoding  : get_current_user(token) – used as a FastAPI dependency
  - WebSocket auth  : verify_ws_token(token)   – called before WS upgrade

For the hackathon a single shared secret is used.
Production would replace this with a proper identity provider.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


class TokenData(BaseModel):
    username: Optional[str] = None
    store_id: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: str


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT token.

    Parameters
    ----------
    data          : payload dict (must include 'sub')
    expires_delta : override the default expiry window
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.jwt_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def decode_token(token: str) -> TokenData:
    """Decode and validate a JWT. Raises HTTPException on failure."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        username: Optional[str] = payload.get("sub")
        store_id: Optional[str] = payload.get("store_id")
        if username is None:
            raise credentials_exception
        return TokenData(username=username, store_id=store_id)
    except JWTError as exc:
        logger.warning("JWT decode failed: %s", exc)
        raise credentials_exception


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
) -> TokenData:
    """FastAPI dependency — injects the current authenticated user."""
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(token)


async def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[TokenData]:
    """Like get_current_user but returns None instead of raising (for mixed endpoints)."""
    if not token:
        return None
    try:
        return decode_token(token)
    except HTTPException:
        return None


def verify_ws_token(token: str) -> TokenData:
    """
    Synchronous WebSocket token verification.
    Called before the WS connection is accepted; raises HTTPException
    on invalid credentials so the connection is rejected with 4008.
    """
    return decode_token(token)
