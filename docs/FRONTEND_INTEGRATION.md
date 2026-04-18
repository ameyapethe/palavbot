# Palavbot Frontend Integration Guide

A self-contained reference for building (or generating) a web frontend that talks to the palavbot backend. Reading this end-to-end should be enough to ship a working UI without access to the backend source.

## 1. What palavbot is

Palavbot is an educational chatbot for an NGO. Users ask questions about breastfeeding and maternal health; the backend runs retrieval-augmented generation (FAISS + OpenAI `gpt-4o-mini`) over a curated corpus of ~40 authoritative sources (ACOG, CDC, WHO, UNICEF, etc.). Answers are grounded in those sources and cited; out-of-scope questions are refused with a fixed message.

**Disclaimers users must see in the UI:**
- Educational content only — not medical advice.
- Not HIPAA compliant — users must not enter PII or PHI.
- Answers may sometimes fall back to the model's general training; when they do, the backend flags it and the UI must mark the message as such.

## 2. TL;DR — what to build

A single-page chat UI:

- Login uses the **existing parent-app Cognito pool** (no new identity service). After login you have a Cognito **ID token** — reuse it.
- On every turn, `POST /chat` with the current message and the full prior conversation. The backend is stateless — it does not remember.
- Render `answer` always. Append a source list if `sources.length > 0`. Append an "external knowledge" footer if `external_knowledge === true`. If `rejected === true`, render just the answer — no decorations.
- Don't persist chat history server-side. You may persist it locally if you want, but the backend won't.

## 3. Environments

Two completely independent backends. Hit **test** during development; switch to **prod** on launch.

| | Test | Prod |
|---|---|---|
| `ApiUrl` | from CloudFormation stack `palavbot-test` → Outputs → `ApiUrl` | from CloudFormation stack `palavbot-prod` → Outputs → `ApiUrl` |
| Cognito User Pool ID | `us-east-1_W3APeJGBp` | `us-east-1_0rlksWAIX` |
| Cognito App Client ID | `6r22psv6mtdrdlkt1835svce7r` | `33o89l0ov50uevb7blbrtsnam3` |
| AWS region | `us-east-1` | `us-east-1` |

The `ApiUrl` looks like `https://<id>.execute-api.us-east-1.amazonaws.com`. Ask the backend operator for the current values, or run:

```bash
aws cloudformation describe-stacks --stack-name palavbot-test \
  --region us-east-1 --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text
```

## 4. Authentication

API Gateway enforces a Cognito JWT authorizer. Every `POST /chat` must include:

```
Authorization: Bearer <Cognito ID token>
```

### Which token to send
**ID token**, not access token. Cognito access tokens don't carry an `aud` claim matching the app client; API Gateway's JWT authorizer rejects them.

### Getting the token
If the parent app is already signing users in with Cognito, reuse its ID token. Common ways to obtain it:

**Amplify (browser):**
```js
import { Auth } from "aws-amplify";
const session = await Auth.currentSession();
const idToken = session.getIdToken().getJwtToken();
```

**`amazon-cognito-identity-js` (browser):**
```js
new CognitoUser({ Username, Pool }).getSession((_err, session) => {
  const idToken = session.getIdToken().getJwtToken();
});
```

**Server-to-server (e.g. tests):**
```js
// AWS SDK v3
import { CognitoIdentityProviderClient, InitiateAuthCommand } from "@aws-sdk/client-cognito-identity-provider";
const client = new CognitoIdentityProviderClient({ region: "us-east-1" });
const r = await client.send(new InitiateAuthCommand({
  ClientId: APP_CLIENT_ID,
  AuthFlow: "USER_PASSWORD_AUTH",
  AuthParameters: { USERNAME, PASSWORD },
}));
const idToken = r.AuthenticationResult.IdToken;
```

### Token lifetime
- Access/ID tokens expire every **60 minutes**.
- Refresh tokens are valid for **30 days**.
- On `401` from palavbot, refresh the session and retry once. Don't retry more than once on the same request.

### What NOT to do
- Don't send the refresh token, client secret, or user password to palavbot. Palavbot only accepts the ID token.
- Don't store JWTs in `localStorage`. Use memory or `sessionStorage`. Amplify handles this for you if you use it.

## 5. Endpoints

### `GET /healthz`
- **Auth:** none
- **Purpose:** liveness check; also useful for warming the Lambda.
- **Response 200:** `{ "ok": true, "chunks": <int> }`
- Safe to poll from anywhere. Don't rely on it for anything user-facing.

### `POST /chat`
- **Auth:** Cognito ID token in `Authorization: Bearer`.
- **Request body:**
  ```json
  {
    "message": "string, 1–4000 chars",
    "history": [
      { "role": "user" | "assistant", "content": "string" }
    ]
  }
  ```
  - `history` is the **full prior transcript**, oldest turn first. Max 40 messages; trim older turns client-side if you exceed that.
  - `role` is only `user` or `assistant`. Never send `system`; the server injects its own system prompt.
  - Trim whitespace but otherwise pass user input through verbatim.

- **Response 200:**
  ```json
  {
    "answer": "string",
    "sources": [
      { "url": "string", "title": "string" }
    ],
    "external_knowledge": false,
    "rejected": false
  }
  ```

### Error responses
| Status | Meaning | UI handling |
|---|---|---|
| `401` | Missing/invalid/expired JWT. | Refresh session, retry once. If still 401, prompt re-login. |
| `403` | JWT valid but wrong audience (e.g. sent prod token to test backend). | Show "session expired, please sign in again". |
| `422` | Malformed body (too long, missing field, bad role). | Treat as bug — log and show a generic error. Don't retry. |
| `429` | API Gateway throttling (rare at our concurrency cap). | Back off 1s, retry once. |
| `503` | Lambda cold-starting / initializing. | Retry after 2s, up to 3 times. |
| `5xx` other | Server error. | Log; show "Something went wrong, please try again." |

## 6. UI rendering rules (**important — backward compatibility**)

The old Streamlit UI inlined these two things into the answer text. The new API returns them as structured fields. The UI must re-render both to preserve the previous user experience.

### 6.1 Always
Render `answer` as markdown. It may contain multi-line text, lists, and inline punctuation. Don't HTML-escape before markdown-rendering, and do sanitize the resulting HTML against XSS.

### 6.2 If `rejected === true`
Render **only** the `answer`. Do not show `sources`, do not show the external-knowledge footer. `answer` will be the fixed string `"I do not have required information. Please try different question"` — you may substitute a more polished localized version if the NGO wants.

### 6.3 If `external_knowledge === true` (and not rejected)
Below the answer, render a subtle visual separator and a note:

```
---
*Note: This topic is not included in the manual; information is being provided from the internet.*
```

Style it as secondary/muted text. Keep the wording; this is the explicit disclosure pattern the NGO approved.

### 6.4 If `sources.length > 0` (and not rejected)
Below the answer (and below the external-knowledge footer if present), render:

```
Additional Resources:
- <link to sources[0].url, labeled with sources[0].title>
- <link to sources[1].url, labeled with sources[1].title>
- …
```

Open external links in a new tab (`target="_blank" rel="noopener noreferrer"`). Sources are already deduplicated server-side; render in order.

### 6.5 Loading state
Per-`POST /chat` latency is typically **2–7 seconds** (OpenAI-bound). Show a typing indicator / spinner. Cold starts add ~3–4 seconds once per ~15 min of inactivity.

### 6.6 Multilingual input
The backend answers in the language of the question (it translates from the English source material). The UI does not need to do language detection. Render whatever the model returns.

## 7. Conversation state

Backend is **stateless**. The client owns the transcript:

1. Maintain an array `[{role, content}, ...]` in component state.
2. On user submit, append `{role: "user", content: <trimmed input>}` to local state, then `POST /chat` with:
   - `message`: the user's new input
   - `history`: the prior messages **excluding** the new one (since the server concatenates them)
3. On 200, append `{role: "assistant", content: response.answer}` to local state and render the response card (with sources / footer per §6).
4. On error, don't append an assistant turn. Show the error inline and keep the user's last input visible so they can retry.

Trim to 40 messages before sending (oldest first). The server rejects larger payloads with 422.

## 8. Reference implementation (React, TypeScript)

Minimal hook, no external deps. Drop into a real app, wrap in your design system.

```tsx
type Role = "user" | "assistant";
type Msg  = { role: Role; content: string };
type Source = { url: string; title: string };
type ChatResponse = {
  answer: string;
  sources: Source[];
  external_knowledge: boolean;
  rejected: boolean;
};

const API_URL = import.meta.env.VITE_PALAVBOT_API_URL; // e.g. https://abc.execute-api…

async function getIdToken(): Promise<string> {
  // Use whatever auth library the parent app uses.
  // Must return a Cognito ID token (not access token).
  throw new Error("wire me up");
}

async function postChat(message: string, history: Msg[]): Promise<ChatResponse> {
  const trimmed = history.slice(-40);
  const token = await getIdToken();
  const r = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`,
    },
    body: JSON.stringify({ message, history: trimmed }),
  });
  if (r.status === 401) {
    // Refresh session once, then retry.
    await refreshSession();
    return postChat(message, history);
  }
  if (!r.ok) throw new Error(`palavbot ${r.status}`);
  return r.json();
}

// In your component:
async function sendMessage(input: string) {
  const user: Msg = { role: "user", content: input.trim() };
  setHistory(h => [...h, user]);
  setStatus("loading");
  try {
    const res = await postChat(user.content, history);
    setHistory(h => [...h, { role: "assistant", content: res.answer }]);
    setLastResponseMeta({
      sources: res.sources,
      external_knowledge: res.external_knowledge,
      rejected: res.rejected,
    });
    setStatus("idle");
  } catch (err) {
    setStatus("error");
  }
}
```

Rendering the assistant bubble (pseudo-JSX):

```tsx
<Markdown>{answer}</Markdown>
{!rejected && external_knowledge && (
  <>
    <hr />
    <p className="muted italic">
      Note: This topic is not included in the manual;
      information is being provided from the internet.
    </p>
  </>
)}
{!rejected && sources.length > 0 && (
  <>
    <p>Additional Resources:</p>
    <ul>
      {sources.map(s =>
        <li key={s.url}>
          <a href={s.url} target="_blank" rel="noopener noreferrer">{s.title}</a>
        </li>
      )}
    </ul>
  </>
)}
```

## 9. CORS

The backend's `Access-Control-Allow-Origin` is controlled by a GitHub repo variable `PALAV_CORS_ORIGIN` in the `test` and `prod` environments. Default is `*`.

Tell the backend operator your production frontend origin (e.g. `https://app.example.org`) so they can lock it down before launch. Until then, calls from any origin work (convenient for dev, not for prod).

## 10. Development workflow

- **No local backend.** There is no local Docker-compose or mock; the backend is designed around Lambda-specific concerns (Lambda Web Adapter, SSM) that aren't worth reproducing locally.
- **Develop against `test`**: set `VITE_PALAVBOT_API_URL` (or equivalent) to the test `ApiUrl`. The test Cognito pool's app client allows `USER_PASSWORD_AUTH` if the backend operator enabled it, letting you script a headless login for tests.
- **Never point dev builds at prod.** The prod Cognito pool issues tokens for real users; rate limits and monitoring are tuned for production traffic.

## 11. Pre-launch checklist

- [ ] Login flow works; ID token captured correctly.
- [ ] Chat round-trip works against test.
- [ ] External-knowledge footer renders when flag is true.
- [ ] Sources render when present; open in new tab.
- [ ] Rejection message renders without decorations.
- [ ] 401 triggers session refresh + retry; persistent 401 triggers re-login.
- [ ] Cold-start delay is covered by the loading state UX.
- [ ] Disclaimer about educational-only / no-PHI is shown prominently on first open.
- [ ] Production CORS origin set on the backend.
- [ ] Switch the frontend to the prod `ApiUrl` + prod pool/client.

## 12. Who to ask

The backend lives at `ameyapethe/palavbot`. File issues there for:
- Bugs in request/response shape.
- New fields you need on `/chat` (e.g. streaming, language override, source re-ranking).
- CORS origin additions.
- Rate-limit increases.

## 13. Known limitations

- **No streaming.** `/chat` is request-response; you won't see tokens arrive progressively. Show a typing indicator instead.
- **No session memory server-side.** If the user refreshes the page, conversation history is lost unless you persist it client-side.
- **English-leaning corpus.** The model translates answers into the user's language, but sources are English. Quality degrades for rare languages.
- **No file upload / image input.** Text only.
- **Single-region (us-east-1).** Latency from other continents is ~100–300ms baseline on top of the OpenAI call.
