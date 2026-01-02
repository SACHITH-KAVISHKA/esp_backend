# Deployment Instructions for Render

## The Fix

The WebSocket timeout issue was caused by using **Gunicorn sync workers** which don't support WebSockets. This has been fixed by:

1. ✅ Switching to **eventlet** async mode
2. ✅ Configuring Gunicorn to use **eventlet workers**
3. ✅ Adding proper timeout configurations

## Deploy to Render

### Option 1: Using Render Dashboard (Recommended)

1. Go to your Render dashboard: https://dashboard.render.com/
2. Find your `safe-speed-api` service
3. Go to **Settings** → **Build & Deploy**
4. Update the **Start Command** to:
   ```bash
   gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 120 --keep-alive 5 --log-level info app:app
   ```
5. Click **Save Changes**
6. The service will automatically redeploy

### Option 2: Automatic Deployment

If you push these changes to GitHub, Render will automatically detect the `Procfile` and use the correct start command.

## What Changed

### Files Modified:

1. **app.py**
   - Added `eventlet.monkey_patch()` at the top (CRITICAL for eventlet to work)
   - Changed SocketIO to use `async_mode='eventlet'`
   - Increased ping_timeout to 120 seconds
   - Removed threading mode

2. **Procfile** (NEW)
   - Specifies the correct Gunicorn command with eventlet workers
   - Sets timeout to 120 seconds for long operations
   - Uses single worker (`-w 1`) as required for SocketIO with eventlet

3. **render.yaml** (NEW - Optional)
   - Blueprint file for automatic Render deployment
   - Can be used for "Infrastructure as Code" deployment

## Verification

After deployment, check the logs for:
- ✅ No more "WORKER TIMEOUT" errors
- ✅ WebSocket connections stay open
- ✅ ESP32 can send data without timeouts
- ✅ Frontend updates in real-time

## Important Notes

⚠️ **CRITICAL**: When using eventlet with SocketIO:
- Must use exactly **1 worker** (`-w 1`)
- Must use `--worker-class eventlet`
- Monkey patch must be at the very top of app.py

## Testing Locally

To test locally before deploying:

```bash
cd esp_backend
pip install -r requirements.txt
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 --timeout 120 app:app
```

Or simply run:
```bash
python app.py
```

## Troubleshooting

If you still see timeout errors:
1. Make sure Render is using the new start command
2. Check that eventlet is installed (should be in requirements.txt)
3. Verify only 1 worker is being used
4. Check Render logs for any import errors
