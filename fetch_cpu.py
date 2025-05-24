import asyncio
import aiohttp

# List endpoint Prometheus tiap cluster
PROMETHEUS_ENDPOINTS = [
    "http://54.88.29.103:30007/api/v1/query",
    "http://54.162.198.32:30007/api/v1/query",
    "http://98.85.113.75:30007/api/v1/query",
]

CLUSTER_DOWN = [
    "http://3.208.173.140:30007/api/v1/query"
]

LOAD ='sum(rate(container_cpu_usage_seconds_total{namespace="nginx", container!="", container!="POD"}[2m])) [30m:2m]'
QUERY = 'sum(rate(node_cpu_seconds_total{mode!="idle"}[2m])) [30m:2m]'

async def fetch_cpu_usage(session, index, url, query):
    async with session.get(f"{url}?query={query}") as response:
        data = await response.json()
        print(data)
        values = [float(r[1]) for r in data["data"]["result"][0]["values"]]
        return index, values  # Nama cluster langsung pakai index

async def fetch_down_cluster_load(session, url, query):
    try:
        async with session.get(f"{url}?query={query}", timeout=5) as response:
            data = await response.json()
            values = [float(r[1]) for r in data["data"]["result"][0]["values"]]
            return {"values": values}
    except Exception as e:
        print(f"Error fetching down cluster data: {e}")
        return {"error": str(e)}

async def get_all_cpu_usage():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_cpu_usage(session, idx, url, QUERY) for idx, url in enumerate(PROMETHEUS_ENDPOINTS)]
        results = await asyncio.gather(*tasks)

    return dict(results)  

async def get_all_cpu_usage_new():
    async with aiohttp.ClientSession() as session:
        # Fetch CPU usage from active clusters
        active_tasks = [fetch_cpu_usage(session, idx, url, QUERY) 
                        for idx, url in enumerate(PROMETHEUS_ENDPOINTS)]
        
        # Fetch CPU usage from down cluster using LOAD query
        down_task = fetch_down_cluster_load(session, CLUSTER_DOWN[0], LOAD)
        
        # Run all tasks concurrently
        results = await asyncio.gather(*active_tasks)
        down_result = await down_task
        
        # Combine results
        all_results = dict(results)
        all_results["down_cluster"] = down_result

    return all_results

if __name__ == "__main__":
    print(asyncio.run(get_all_cpu_usage()))
