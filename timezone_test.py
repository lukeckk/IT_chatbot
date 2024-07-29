from datetime import datetime
import pytz

# Get current date and time
now = datetime.now()
pacific = pytz.timezone('America/Los_Angeles')
now_pacific = datetime.now(pacific)
time = now_pacific.strftime("%Y-%m-%d %H:%M")

print(time)