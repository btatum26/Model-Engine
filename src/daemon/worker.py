import os
import redis
from rq import Worker, Queue

# Setup Redis connection
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
conn = redis.from_url(redis_url)

if __name__ == '__main__':
    print("Starting RQ Worker... Listening to 'default' queue.")
    
    # In RQ 2.0.0+, we pass the connection explicitly instead of using `with Connection():`
    queue = Queue('default', connection=conn)
    worker = Worker([queue], connection=conn)
    
    # Start consuming tasks
    worker.work()