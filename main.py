"""
We are given a list of cities and a number of days in which to plan a round-trip vacation visiting all of them.
For each pair of cities, and for every date in our range, we have a cost to travel from city A to city B.
We need to find the optimal itinerary to minimize the total flight cost.
"""
import numpy as np
from typing import List
from pydantic import BaseModel, PositiveInt, model_validator
from ortools.sat.python.cp_model import CpModel, CpSolver


class TravelInstance(BaseModel):
    flight_costs: np.ndarray  # dim: num days x source city x destination city
    city_names: List[str]

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def validate_dimensions(cls, v):
        d, source, dest = v.flight_costs.shape

        if d < source - 1:
            raise ValueError("Number of unique cities exceeds number of days, round trip impossible")
        if len(v.city_names) != source:
            raise ValueError("Number of city names does not match number of cities")
        if source != dest:
            raise ValueError("Number of source cities does not match number of destination cities")
        return v


if __name__ == "__main__":
    cities = ["Houston", "Austin", "Dallas", "Los Angeles", "New York", "Chicago"]
    instance = TravelInstance(flight_costs=np.random.uniform(50, 1000, size=(10, len(cities), len(cities))),
                              city_names=cities)
    print(instance)
