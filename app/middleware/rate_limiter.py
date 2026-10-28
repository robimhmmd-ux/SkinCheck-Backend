import time
from typing import Dict, Tuple
from fastapi import Request, HTTPException

# Simple in-memory rate limiter using Token Bucket / Sliding window approach
# In a real production app, this would use Redis.
_RATE_LIMITS: Dict[str, Tuple[int, float]] = {}

def check_rate_limit(request: Request, limit: int = 10, window: int = 60):
    """
    Dependency to check if the client IP has exceeded the rate limit.
    limit: max requests
    window: time window in seconds
    """
    client_ip = request.client.host if request.client else "127.0.0.1"
    path = request.url.path
    key = f"{client_ip}:{path}"
    
    current_time = time.time()
    
    if key in _RATE_LIMITS:
        requests, window_start = _RATE_LIMITS[key]
        if current_time - window_start > window:
            # Reset window
            _RATE_LIMITS[key] = (1, current_time)
        else:
            if requests >= limit:
                raise HTTPException(status_code=429, detail="Terlalu banyak permintaan (Too Many Requests). Harap tunggu beberapa saat.")
            _RATE_LIMITS[key] = (requests + 1, window_start)
    else:
        _RATE_LIMITS[key] = (1, current_time)
        
def rate_limit(limit: int = 10, window: int = 60):
    """Closure to pass arguments to the dependency."""
    def _rate_limit(request: Request):
        check_rate_limit(request, limit, window)
    return _rate_limit
