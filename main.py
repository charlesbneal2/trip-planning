"""
We are given a list of cities and a number of days in which to plan a round-trip vacation visiting all of them.
For each pair of cities, and for every date in our range, we have a cost to travel from city A to city B.
We need to find the optimal itinerary to minimize the total flight cost.

Constraints:
- for each day, no more than one flight can be picked
- on the last day, only trips to the origin (0) can be picked
- each node must only be visited once
"""
import numpy as np
from typing import List
from pydantic import BaseModel, PositiveInt, model_validator
from ortools.sat.python.cp_model import CpModel, CpSolver, OPTIMAL, FEASIBLE


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


def solve_travel(travel: TravelInstance):
    model = CpModel()
    solver = CpSolver()

    n_days, n_cities, _ = travel.flight_costs.shape
    flights = {(d, i, j): model.new_bool_var(f"flight_{d}_{i}_{j}") for (d, i, j), _ in
               np.ndenumerate(travel.flight_costs)
               if i != j}

    city = [model.new_int_var(0, n_cities - 1, f'city_{d}') for d in range(n_days + 1)]

    for i in range(1, n_cities):
        model.add(sum(flights[d, i, j] for d in range(n_days) for j in range(n_cities) if i != j) == 1)
        model.add(sum(flights[d, j, i] for d in range(n_days) for j in range(n_cities) if i != j) == 1)

    for d in range(n_days):
        model.add(sum(flights[d, i, j] for i in range(n_cities) for j in range(n_cities) if i != j) <= 1)
    model.add(sum(flights.values()) == n_cities)

    for (d, i, j), _ in np.ndenumerate(travel.flight_costs):
        if i == j:
            continue
        model.add(city[d + 1] == j).only_enforce_if(flights[d, i, j])
        model.add(city[d + 1] != j).only_enforce_if(flights[d, i, j].Not())

    model.add(city[0] == 0)
    model.add(city[n_days] == 0)

    model.minimize(sum(flights[d, i, j] * travel.flight_costs[d, i, j] for (d, i, j), _ in flights.items()))

    status = solver.solve(model)

    if status == OPTIMAL:
        print("Optimal solution found!")
    elif status == FEASIBLE:
        print("Feasible solution found!")
    else:
        print("No solution found.")

    if status in (OPTIMAL, FEASIBLE):
        tour = [f"Day {d}: {travel.city_names[i]} to {travel.city_names[j]} with cost {travel.flight_costs[d, i, j]}"
                for (d, i, j), _ in flights.items() if solver.value(flights[d, i, j])]
        print("\n".join(tour))
    else:
        print("Solver status:", solver.StatusName(status))


if __name__ == "__main__":
    cities = ["Houston", "Austin", "Dallas", "Los Angeles", "New York", "Chicago"]
    instance = TravelInstance(flight_costs=np.random.uniform(50, 1000, size=(10, len(cities), len(cities))),
                              city_names=cities)
    solve_travel(instance)

