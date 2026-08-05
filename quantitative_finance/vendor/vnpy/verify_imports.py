import sys
import vnpy

from vnpy.event import Event, EventEngine
from vnpy.trader.constant import Exchange, Direction, OrderType
from vnpy.trader.object import SubscribeRequest, OrderRequest

print("python", sys.version)
print("vnpy", getattr(vnpy, "__version__", "unknown"))
print("imports_ok")
print(Exchange.SSE.value, Direction.LONG.value, OrderType.LIMIT.value)