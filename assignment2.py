"""
FIT2004 Assignment 2
Author: Emily Samantha Zarry
ID: 32558945

"""
from queue import PriorityQueue


# Part 1
class RoadGraph:
    """
    Class that represents the road network in the city.

    """

    def __init__(self, roads: list, cafes: list) -> None:
        """
        Constructor of RoadGraph class.

        :Input:
            roads: list of roads represented as tuples (u, v, w), where
                    u is the starting location ID for a road,
                    v is the ending location ID for a road,
                    w is the time taken to travel from a location u to a location v
            cafes: list of cafes represented as tuples (location, waiting_time), where
                    location is the location of the cafe,
                    waiting_time is the waiting time for a coffee in the cafe

        :Output: None

        :Time Complexity: Need to be O(V + E)
        :Aux Space Complexity: Need to be O(V + E)
        """
        self.cafes = cafes

        # Determining the number of locations (number of vertices)
        self.num_locations = 0
        for i in range(len(roads)):  # O(E)
            if roads[i][0] > self.num_locations:  # Comparing the starting location ID
                self.num_locations = roads[i][0]
            if roads[i][1] > self.num_locations:  # Comparing the ending location ID
                self.num_locations = roads[i][1]

        # Initializing adjacency list
        self.graph = []
        self.fill_adjacency_list(self.graph, roads, cafes)

        # Initializing adjacency list of reversed roads
        self.graph_reversed = []
        self.fill_adjacency_list(self.graph_reversed, roads, cafes, True)

    def routing(self, start: int, end: int) -> list:
        """
        Calculates the optimal routes for commuting while grabbing coffee along the way.

        :Input:
            start: starting location of the journey
            end: ending location of the journey

        :Output: the shortest route from the start location to the end location,
                 going through at least 1 of the locations listed in cafes,
                 if no such route exists, returns None

        :Time Complexity: Need to be O(E log V)
        :Aux Space Complexity: Need to be O(V + E)
        """
        # Apply Dijkstra's algorithm with the start location as source
        dist_start, pred_start = dijkstra(self.graph, start)

        # Apply Dijkstra's algorithm with the ending location as source
        # We pass the graph with the roads reversed to maintain the reachability
        dist_end, pred_end = dijkstra(self.graph_reversed, end)

        # Get the quickest time from start to a cafe and from that cafe to the end
        min_time, cafe_to_visit = self.get_min(dist_start, dist_end)

        return self.construct_path(cafe_to_visit, pred_start, pred_end)

    def fill_adjacency_list(self, graph: list, roads: list, cafes: list, reverse=False) -> None:
        """
        Method to fill an adjacency list given the list of roads and cafes.

        :Input:
            graph: adjacency list to be filled
            roads: list of roads represented as tuples (u, v, w), where
                    u is the starting location ID for a road,
                    v is the ending location ID for a road,
                    w is the time taken to travel from a location u to a location v
            cafes: list of cafes represented as tuples (location, waiting_time), where
                    location is the location of the cafe,
                    waiting_time is the waiting time for a coffee in the cafe
            reverse: boolean value to indicate whether the road direction should be reversed

        :Post-condition: the graph is an adjacency list of the given values

        :Time Complexity: O(max(V, E))
        :Aux Space Complexity: O(1)
        """
        for i in range(self.num_locations + 1):  # O(V)
            # The first element of each list is reserved to indicate whether the location has a cafe.
            # If yes, it will store the waiting time. If not, it will store None.
            graph.append([None])

        # Adding cafe locations and the waiting time
        for i in range(len(cafes)):  # O(V)
            cafe_location, waiting_time = cafes[i]
            graph[cafe_location][0] = waiting_time

        # Filling the adjacency list
        for i in range(len(roads)):  # O(E)
            start_location, end_location, travel_time = roads[i]
            if not reverse:
                graph[start_location].append((end_location, travel_time))
            else:
                graph[end_location].append((start_location, travel_time))

    def get_min(self, dist_start: list, dist_end: list) -> int:
        """
        Method to get the minimum travel time from start to end, while visiting a cafe along the way.

        :Input:
            dist_start: list of minimum distance to each location ID, with start location as source
            dist_end: list of minimum distance to each location ID, with ending location as source

        :Output:
            min_time: the minimum travel time
            cafe_to_visit: the location ID of the cafe to be visited

        :Time Complexity: O(V)
        :Aux Space Complexity: O(1)
        """
        # Initialize min_time as infinity and cafe_to_Visit as None
        min_time = float('inf')
        cafe_to_visit = None

        # For each cafe location, calculate the minimum travel time for visiting that cafe
        for cafe_location, cafe_time in self.cafes:
            # If the minimum travel time for visiting that cafe is less than the current min_time
            if dist_start[cafe_location] + cafe_time + dist_end[cafe_location] < min_time:
                # Set as min_time
                min_time = dist_start[cafe_location] + cafe_time + dist_end[cafe_location]
                # Record that cafe as cafe_to_visit
                cafe_to_visit = cafe_location

        return min_time, cafe_to_visit

    def construct_path(self, cafe_to_visit: int, pred_start: list, pred_end: list) -> list:
        """
        Method to construct the path with minimum time from start to end,
        while visiting a cafe along the way.

        :Input:
            cafe_to_visit: location ID of the cafe to be visited
            pred_start: list of predecessor of each location ID in the minimum path,
                        with the start location as source
            pred_end: list of predecessor of each location ID in the minimum path,
                      with the ending location as source

        :Output:
            path: list of location IDs to visit in the path with minimum time

        :Time Complexity: O(V)
        :Aux Space Complexity: O(V)
        """
        path = []

        # Getting the path from start to cafe
        prev_location = pred_start[cafe_to_visit]
        while prev_location is not None:
            path.append(prev_location)
            prev_location = pred_start[prev_location]

        path.reverse()  # Reverse the path
        path.append(cafe_to_visit)

        # Getting the path from cafe to end
        prev_location = pred_end[cafe_to_visit]
        while prev_location is not None:
            path.append(prev_location)
            prev_location = pred_end[prev_location]

        return path


def dijkstra(graph: list, start: int) -> list:
    """
    Function for implementing Dijkstra's algorithm.

    :Input:
        graph: adjacency list that represents the road network in the city
        start: start location ID

    :Output:
        dist: list of minimum distance to each location in the graph, with start as source
        pred: list of predecessor location of each location that gives the minimum distance

    :Time complexity: O(E log V)
    :Aux Space Complexity: O(V)
    """
    num_locations = len(graph)

    # Initializing distance array to infinity
    dist = [float('inf')] * num_locations  # O(V)

    # Initializing predecessor array
    pred = [None] * num_locations  # O(V)

    # Set distance of the starting location to 0
    dist[start] = 0

    # Initialize priority queue
    queue = PriorityQueue()
    queue.put((0, start))

    # Main loop
    while not queue.empty():  # O(V)
        # Extracting the nearest location u
        dist_u, u = queue.get()

        # Only processing up-to-date entries
        if dist_u == dist[u]:
            # For each neighbour v of u, relax along that edge
            for i in range(1, len(graph[u])):  # Skipping the first element (Waiting time)
                v, time = graph[u][i]

                # If distance of u plus travel time of that road is quicker than the current distance of v
                if dist_u + time < dist[v]:
                    # Update the distance of v
                    dist[v] = dist[u] + time
                    # Set the predecessor of v as u
                    pred[v] = u
                    # Update the priority queue
                    queue.put((dist[v], v))

    return dist, pred


# TESTING TASK 1
roads = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2),
         (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2),
         (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
cafes = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]

mygraph = RoadGraph(roads, cafes)
print(mygraph.routing(3, 4))


# Part 2
def optimalRoute(downhillScores: list, start: int, finish: int) -> list:
    """
    Calculates the route that should be used for going from the starting point to the
    finishing point while using only downhill segments and obtaining the maximum score.

    :Input:
        downhillScores: list of downhill segments represented as tuples (a, b, c), where
            a is the start point of a downhill segment,
            b is the end point of a downhill segment
            c is the integer score for using this downhill segment to go from point a to b
        start: the start point of the tournament
        finish: the end point of the tournament

    Output: the optimal route for obtaining the maximum score

    :Time Complexity: Need to be O(D), where D is the number of downhill segments (edges)
    :Aux Space Complexity:
    """
