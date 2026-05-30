"""
src/alerts/twilio_alert.py
───────────────────────────
Twilio-based alert system. Sends SMS and optionally a voice call when
the anomaly detector triggers a critical-severity security event.

When Twilio credentials are absent (dev environment), falls back to
structured console logging so the alert cascade can still be tested.

Mock sandbox testing:
  Point TWILIO_BASE_URL at a Stoplight Prism server running the
  Twilio OpenAPI spec to validate outbound requests without hitting
  the live network or dispatching real calls.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class AlertSystem:
    """
    Fires Twilio SMS + voice alerts on critical security events.

    Parameters
    ----------
    settings : Settings
        Must contain twilio_account_sid, twilio_auth_token,
        twilio_from_number, alert_phone_number.
    base_url : str
        Override the Twilio API base URL (use for mock sandbox testing).
    """

    TWILIO_BASE = "https://api.twilio.com/2010-04-01"

    def __init__(self, settings, base_url: Optional[str] = None) -> None:
        self.settings   = settings
        self._base_url  = base_url or os.environ.get("TWILIO_BASE_URL", self.TWILIO_BASE)
        self._client    = None
        self._sdk_loaded = False

        self._account_sid  = getattr(settings, "twilio_account_sid", None)
        self._auth_token   = getattr(settings, "twilio_auth_token", None)
        self._from_number  = getattr(settings, "twilio_from_number", None)
        self._to_number    = getattr(settings, "alert_phone_number", None)

        if all([self._account_sid, self._auth_token, self._from_number, self._to_number]):
            logger.info("AlertSystem: Twilio credentials loaded (to=%s)", self._to_number)
        else:
            logger.warning(
                "AlertSystem: Twilio credentials not set — alerts will be console-only. "
                "Set SIS_TWILIO_ACCOUNT_SID, SIS_TWILIO_AUTH_TOKEN, "
                "SIS_TWILIO_FROM_NUMBER, SIS_ALERT_PHONE_NUMBER."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def send_alert(
        self,
        store_id: str,
        person_id: int,
        anomaly_score: float,
        alert_type: str = "shoplifting",
        keyframe_url: Optional[str] = None,
    ) -> bool:
        """
        Send SMS (always) + voice call (only for score > 0.90).

        Returns
        -------
        True if at least the SMS was dispatched successfully.
        """
        ts = datetime.utcnow().strftime("%H:%M:%S UTC")
        message = (
            f"[SIS ALERT] {ts} | Store: {store_id} | "
            f"Type: {alert_type.upper()} | "
            f"Person ID: {person_id} | Score: {anomaly_score:.2f}"
        )
        if keyframe_url:
            message += f" | Evidence: {keyframe_url}"

        sms_ok = self.send_sms(message)

        if anomaly_score > 0.90:
            tts_message = (
                f"Security alert at {store_id}. Suspicious behaviour detected. "
                f"Anomaly confidence {int(anomaly_score * 100)} percent. "
                "Please review the security feed immediately."
            )
            self.make_voice_call(tts_message)

        return sms_ok

    def send_sms(self, message: str) -> bool:
        """Send an SMS via Twilio Messaging API."""
        if not self._has_credentials():
            logger.warning("[ALERT SMS] %s", message)
            return False

        try:
            client = self._get_client()
            msg = client.messages.create(
                body=message,
                from_=self._from_number,
                to=self._to_number,
            )
            logger.info("SMS sent: SID=%s status=%s", msg.sid, msg.status)
            return True
        except Exception as exc:
            logger.error("SMS failed: %s", exc)
            # Try raw httpx fallback (avoids SDK import issues)
            return self._send_sms_raw(message)

    def make_voice_call(self, tts_message: str) -> bool:
        """Initiate a voice call with TTS message via Twilio Voice API."""
        if not self._has_credentials():
            logger.warning("[ALERT CALL] %s", tts_message)
            return False

        try:
            client = self._get_client()
            twiml = f"<Response><Say>{tts_message}</Say></Response>"
            call = client.calls.create(
                twiml=twiml,
                from_=self._from_number,
                to=self._to_number,
            )
            logger.info("Voice call initiated: SID=%s status=%s", call.sid, call.status)
            return True
        except Exception as exc:
            logger.error("Voice call failed: %s", exc)
            return False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _has_credentials(self) -> bool:
        return all([self._account_sid, self._auth_token, self._from_number, self._to_number])

    def _get_client(self):
        if self._client is None:
            from twilio.rest import Client
            self._client = Client(self._account_sid, self._auth_token)
        return self._client

    def _send_sms_raw(self, message: str) -> bool:
        """Fallback: raw httpx POST to Twilio Messages endpoint."""
        try:
            import httpx
            url = f"{self._base_url}/Accounts/{self._account_sid}/Messages.json"
            response = httpx.post(
                url,
                auth=(self._account_sid, self._auth_token),
                data={"From": self._from_number, "To": self._to_number, "Body": message},
                timeout=10.0,
            )
            response.raise_for_status()
            logger.info("SMS sent via raw httpx: %s", response.json().get("sid"))
            return True
        except Exception as exc:
            logger.error("Raw SMS also failed: %s", exc)
            return False


class MockAlertSystem(AlertSystem):
    """
    Drop-in replacement for CI/CD pipelines or dev environments.
    Logs all alert calls but never touches the Twilio network.
    """

    def __init__(self, *args, **kwargs) -> None:
        logger.info("MockAlertSystem initialised — no real Twilio calls will be made")
        self._alerts_fired: list[dict] = []

    def send_sms(self, message: str) -> bool:
        logger.info("[MOCK SMS] %s", message)
        self._alerts_fired.append({"type": "sms", "message": message})
        return True

    def make_voice_call(self, tts_message: str) -> bool:
        logger.info("[MOCK CALL] %s", tts_message)
        self._alerts_fired.append({"type": "call", "message": tts_message})
        return True

    def send_alert(self, store_id, person_id, anomaly_score, alert_type="shoplifting", keyframe_url=None) -> bool:
        logger.warning("[MOCK ALERT] store=%s person=%d score=%.2f type=%s",
                       store_id, person_id, anomaly_score, alert_type)
        return True

    def get_fired_alerts(self) -> list[dict]:
        return list(self._alerts_fired)
