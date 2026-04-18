"""Post-deploy smoke tests. Skipped unless PALAV_API_URL is set.

The CI `deploy.yml` workflow exports the stack outputs as env vars and runs
this module after `sam deploy` succeeds. A Cognito user pre-created once
(credentials in GitHub secrets) is used to obtain a real JWT.
"""

from __future__ import annotations

import os

import httpx
import pytest

API_URL = os.environ.get("PALAV_API_URL")
USER_POOL_ID = os.environ.get("PALAV_USER_POOL_ID")
CLIENT_ID = os.environ.get("PALAV_CLIENT_ID")
TEST_USER = os.environ.get("PALAV_TEST_USER")
TEST_PASSWORD = os.environ.get("PALAV_TEST_PASSWORD")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

pytestmark = pytest.mark.skipif(
    not all([API_URL, USER_POOL_ID, CLIENT_ID, TEST_USER, TEST_PASSWORD]),
    reason="Smoke-test env vars not set; running outside a deployed environment.",
)


def _get_jwt() -> str:
    import boto3

    cognito = boto3.client("cognito-idp", region_name=AWS_REGION)
    resp = cognito.initiate_auth(
        ClientId=CLIENT_ID,
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={"USERNAME": TEST_USER, "PASSWORD": TEST_PASSWORD},
    )
    return resp["AuthenticationResult"]["IdToken"]


def test_unauth_request_rejected():
    r = httpx.post(f"{API_URL}/chat", json={"message": "hi", "history": []}, timeout=10)
    assert r.status_code in (401, 403)


def test_healthz_open():
    r = httpx.get(f"{API_URL}/healthz", timeout=10)
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_authenticated_chat_answers_breastfeeding_question():
    token = _get_jwt()
    r = httpx.post(
        f"{API_URL}/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"message": "how often should I nurse a newborn?", "history": []},
        timeout=30,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"]
    assert body["rejected"] is False
