"""
Stage B - green duration from queue clearance.

    clearance = queue / (saturation_flow - arrival_rate)
    G = clip(clearance, MIN_GREEN, MAX_GREEN)

Vehicles that arrive while the light is green are accounted for by
subtracting the arrival rate from the discharge rate.
"""

from . import config as C


def clip_green(g):
    return int(max(C.MIN_GREEN, min(C.MAX_GREEN, round(g))))


def queue_clearance_green(queue, direction, time_of_day="NORMAL"):
    arrival = C.ARRIVAL_RATES.get(time_of_day, C.ARRIVAL_RATES["NORMAL"])[direction]
    net_flow = max(C.SATURATION_FLOW - arrival, 0.1)
    return clip_green(queue / net_flow)
