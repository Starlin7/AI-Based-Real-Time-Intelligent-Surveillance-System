"""
════════════════════════════════════════════════════════════════════════════════
sendmail.py — Email Alert Module for Real-Time Surveillance System
════════════════════════════════════════════════════════════════════════════════

Sends HTML-formatted email alerts with optional screenshot attachment.
Always called from a daemon thread — never invoked on the main thread.

SMTP credentials are read from environment variables:
    SMTP_HOST        (default: smtp.gmail.com)
    SMTP_PORT        (default: 587)
    SMTP_USER        (required)
    SMTP_PASSWORD    (required)
    ALERT_RECIPIENT  (required)
"""

import logging
import os
import smtplib
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

log = logging.getLogger(__name__)


def sendalert(image_path: str, message: str, threat_level: str = "HIGH") -> None:
    """Send an HTML email alert with an optional screenshot attachment.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the screenshot JPEG.  If the file does
        not exist the email is still sent (without an attachment).
    message : str
        Plain-text description of the event that triggered the alert.
    threat_level : str
        ``"CRITICAL"`` or ``"HIGH"`` — determines subject line colour and
        header styling.
    """
    try:
        smtp_host: str = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        smtp_port: int = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user: str | None = os.environ.get("SMTP_USER")
        smtp_pass: str | None = os.environ.get("SMTP_PASSWORD")
        recipient: str | None = os.environ.get("ALERT_RECIPIENT")

        if not all([smtp_user, smtp_pass, recipient]):
            log.warning(
                "Email alert skipped — SMTP_USER, SMTP_PASSWORD, or "
                "ALERT_RECIPIENT environment variable not set."
            )
            return

        # ── Subject line ──────────────────────────────────────────────────
        subject_tag = (
            "[CRITICAL ALERT]" if threat_level == "CRITICAL" else "[HIGH ALERT]"
        )
        subject = f"{subject_tag} Surveillance System — {message[:80]}"

        # ── HTML body ─────────────────────────────────────────────────────
        header_gradient = (
            "linear-gradient(135deg, #b91c1c 0%, #dc2626 50%, #ef4444 100%)"
            if threat_level == "CRITICAL"
            else "linear-gradient(135deg, #c2410c 0%, #ea580c 50%, #f97316 100%)"
        )
        badge_bg = "#fef2f2" if threat_level == "CRITICAL" else "#fff7ed"
        badge_text = "#991b1b" if threat_level == "CRITICAL" else "#9a3412"
        badge_border = "#fecaca" if threat_level == "CRITICAL" else "#fed7aa"
        icon = "&#128680;" if threat_level == "CRITICAL" else "&#9888;&#65039;"
        timestamp = time.strftime("%B %d, %Y at %I:%M:%S %p")

        html_body = f"""\
<html>
<body style="margin: 0; padding: 0; background: #111827; font-family: 'Segoe UI', Arial, sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background: #111827; padding: 32px 16px;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background: #1f2937; border-radius: 16px; overflow: hidden; box-shadow: 0 20px 60px rgba(0,0,0,0.5);">

        <!-- Header -->
        <tr>
          <td style="background: {header_gradient}; padding: 32px 32px 28px 32px; text-align: center;">
            <div style="font-size: 48px; margin-bottom: 8px;">{icon}</div>
            <h1 style="color: #ffffff; margin: 0; font-size: 24px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;">
              {threat_level} THREAT DETECTED
            </h1>
            <p style="color: rgba(255,255,255,0.8); margin: 8px 0 0; font-size: 13px; letter-spacing: 0.5px;">
              Real-Time Intelligent Surveillance System
            </p>
          </td>
        </tr>

        <!-- Threat Badge -->
        <tr>
          <td style="padding: 28px 32px 0 32px;">
            <table width="100%" cellpadding="0" cellspacing="0" style="background: {badge_bg}; border: 1px solid {badge_border}; border-radius: 10px;">
              <tr>
                <td style="padding: 16px 20px;">
                  <span style="display: inline-block; background: {badge_text}; color: #ffffff; font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 20px; letter-spacing: 1px; text-transform: uppercase;">
                    {threat_level}
                  </span>
                  <p style="color: {badge_text}; font-size: 14px; margin: 10px 0 0; font-weight: 600; line-height: 1.5;">
                    {message}
                  </p>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Details -->
        <tr>
          <td style="padding: 24px 32px;">
            <!-- Timestamp -->
            <table width="100%" cellpadding="0" cellspacing="0" style="background: #374151; border-radius: 10px; margin-bottom: 16px;">
              <tr>
                <td style="padding: 14px 20px;">
                  <span style="color: #9ca3af; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">&#128339; Detected At</span>
                  <p style="color: #e5e7eb; font-size: 15px; margin: 6px 0 0; font-weight: 500;">{timestamp}</p>
                </td>
              </tr>
            </table>

            <!-- Action Required -->
            <table width="100%" cellpadding="0" cellspacing="0" style="background: #374151; border-radius: 10px;">
              <tr>
                <td style="padding: 14px 20px;">
                  <span style="color: #9ca3af; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">&#128270; Action Required</span>
                  <p style="color: #e5e7eb; font-size: 14px; margin: 6px 0 0; line-height: 1.6;">
                    Please review the attached screenshot and take appropriate action immediately.
                    This alert was generated automatically by the AI-powered threat analysis engine.
                  </p>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background: #111827; padding: 20px 32px; text-align: center; border-top: 1px solid #374151;">
            <p style="color: #6b7280; font-size: 11px; margin: 0; line-height: 1.6;">
              &#128274; This is an automated security alert from the<br>
              <strong style="color: #9ca3af;">Real-Time Intelligent Surveillance System</strong><br>
              Context-Aware Multi-Condition Threat Assessment
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""

        # ── Compose MIME message ──────────────────────────────────────────
        msg = MIMEMultipart()
        msg["From"] = smtp_user  # type: ignore[assignment]
        msg["To"] = recipient  # type: ignore[assignment]
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html"))

        # Attach screenshot if the file exists
        img_path = Path(image_path)
        if img_path.is_file():
            with open(img_path, "rb") as fp:
                img_data = fp.read()
            img_attachment = MIMEImage(img_data, name=img_path.name)
            img_attachment.add_header(
                "Content-Disposition", "attachment", filename=img_path.name
            )
            msg.attach(img_attachment)
            log.info("Screenshot attached: %s", img_path.name)
        else:
            log.warning("Screenshot file not found, sending without attachment: %s", image_path)

        # ── Send via SMTP (TLS) ──────────────────────────────────────────
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_pass)  # type: ignore[arg-type]
            server.sendmail(smtp_user, [recipient], msg.as_string())  # type: ignore[arg-type]

        log.info("Email alert sent → %s  [%s]", recipient, subject)

    except smtplib.SMTPAuthenticationError:
        log.error("SMTP authentication failed — check SMTP_USER / SMTP_PASSWORD.")
    except smtplib.SMTPException as exc:
        log.error("SMTP error while sending alert email: %s", exc)
    except Exception:
        log.exception("Unexpected error in sendalert()")
