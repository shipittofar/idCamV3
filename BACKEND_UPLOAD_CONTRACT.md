# Backend Upload Contract

Source of truth: `src/idcam/network/api_client.py` (`ApiClient.upload_document`).  
This document describes **only what the client sends** after a capture file exists. It is not an API specification invented for the backend.

---

## 1. Backend Endpoint

| Item | Current implementation |
|------|------------------------|
| HTTP method | **POST** |
| Scheme | Taken from `server.url` (`pydantic.HttpUrl`). Deployed configs use **HTTPS** (`https://…`). The client does not force the scheme in code. |
| Path | `server.path`, default **`/api/document`**. If the configured path has no leading `/`, one is prepended. |
| Query string | **None.** The URL is path-only. |

**URL construction** (`api_client.py`):

```text
url = str(config.server.url).rstrip("/") + config.server.path
```

**Runtime sources:**

- `config.server.url` — YAML `server.url`, or environment **`IDCAM_SERVER_URL`** if set.
- `config.server.path` — YAML `server.path`, else **`/api/document`**.

**Example** (illustrative; host is site-specific):

```text
POST https://<host>/api/document
```

If `server.url` is `https://uploads.example.net/` and `server.path` is `/api/document`, the request URL is:

```text
https://uploads.example.net/api/document
```

There is no other backend image-upload call in this client.

---

## 2. Request Headers

### Set explicitly by the application

| Header | Always sent? | Value |
|--------|----------------|--------|
| `X-Device-ID` | Yes | `config.device_id` (YAML `device_id`, or **`IDCAM_DEVICE_ID`** if set) |
| `X-Capture-ID` | Yes | Capture id: `cap-` + 32 hex characters (`uuid4`) |
| `Idempotency-Key` | Yes | **Same string as** `X-Capture-ID` |
| `Authorization` | **Only if a token is resolved** | `Bearer <token>` |

**Token resolution** (`AuthConfig.resolved_token`):

- Sent when `auth.type` is `bearer` **and** `auth.token` is non-empty.
- Token comes from **`IDCAM_AUTH_TOKEN`** (overrides YAML and forces `auth.type=bearer`) or from YAML `auth.token`.
- If `auth.type` is `none` or the token is empty, **`Authorization` is omitted**.

No API-key header other than this optional Bearer token is set.

### Not set by application code

| Header | Status |
|--------|--------|
| `Accept` | **Not set** |
| `Content-Type` | **Not set in the `headers` dict.** httpx sets it because `files=` is used (see below). |
| `X-Api-Key` / similar | **Not sent** |

### Set by httpx (not customized in this repo)

Because the client uses `httpx.AsyncClient.post(..., files=...)` with no extra `headers` beyond the table above, the library also sends typical HTTP headers, including:

- `Host` — from the URL
- `Content-Type` — **`multipart/form-data; boundary=<generated>`**
- `Content-Length` (or equivalent transfer encoding)
- `User-Agent` — httpx default (e.g. `python-httpx/<version>`), not overridden

TLS: `verify=config.server.verify_tls` (default `true`). Redirects are not followed (`follow_redirects=False`).

---

## 3. Request Body / Payload

**Format: `multipart/form-data`**

Not used:

- JSON body
- raw binary POST (no multipart)
- base64 in a JSON/text field
- query parameters

**Call:**

```python
files = {"image": (image_path.name, image_file, "image/jpeg")}
self._client.post(url, headers=headers, files=files)
```

The file is opened as **raw bytes** (`open(..., "rb")`) and attached as a multipart file part.

### Multipart fields

| Field name | Kind | Contents |
|------------|------|----------|
| **`image`** | File (only field) | JPEG bytes of the captured still |

**There are no other form fields.** Device id, capture id, timestamp, SHA-256, dimensions, and `created_at` are **not** in the body.

`UploadRequest.created_at` is filled in the orchestrator but **never copied into headers or multipart**.

### Image part details

| Item | Value |
|------|--------|
| Field name | `image` |
| Filename | `image_path.name` — production captures are named **`capture-<32 hex>.jpg`**. This UUID is **independent of** `X-Capture-ID`. |
| MIME type on the part | `image/jpeg` |
| Image format | JPEG (still written as JPEG; declared as `image/jpeg`) |
| Encoding | Binary JPEG in the multipart part (not base64) |

### Representative request

```http
POST /api/document HTTP/1.1
Host: <from server.url>
X-Device-ID: <device_id>
X-Capture-ID: cap-<32 hex>
Idempotency-Key: cap-<32 hex>
Authorization: Bearer <redacted>    # omitted if no token
Content-Type: multipart/form-data; boundary=<httpx-boundary>
User-Agent: python-httpx/<version>

--<httpx-boundary>
Content-Disposition: form-data; name="image"; filename="capture-<other-32-hex>.jpg"
Content-Type: image/jpeg

<raw JPEG bytes>
--<httpx-boundary>--
```

**STATIC STRUCTURE:** method, path default, header names, single multipart field `image`, JPEG MIME.

**RUNTIME VALUES:** host, `device_id`, `capture_id`, optional Bearer token, JPEG bytes, filenames, httpx boundary and `User-Agent`.
