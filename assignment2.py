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
    def __init__(self, roads, cafes):
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
        for i in range(len(roads)):                 # O(E)
            if roads[i][0] > self.num_locations:    # Comparing the starting location ID
                self.num_locations = roads[i][0]
            if roads[i][1] > self.num_locations:    # Comparing the ending location ID
                self.num_locations = roads[i][1]

        # Initializing adjacency list
        self.graph = []
        self.fill_adjacency_list(self.graph, roads, cafes)

        # Initializing adjacency list of reversed roads
        self.graph_reversed = []
        self.fill_adjacency_list(self.graph_reversed, roads, cafes, True)

    def fill_adjacency_list(self, graph, roads, cafes, reverse=False):
        """

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

    def routing(self, start, end):
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
        # We pass the graph with the roads reversed
        dist_end, pred_end = dijkstra(self.graph_reversed, end)

        # print(dist_start)
        # print(pred_start)
        # print(dist_end)
        # print(pred_end)

        # Get the quickest time from start to a cafe and from that cafe to the end
        min_time = float('inf')
        cafe_to_visit = None
        for cafe_location, cafe_time in self.cafes:
            if dist_start[cafe_location] + cafe_time + dist_end[cafe_location] < min_time:
                min_time = dist_start[cafe_location] + cafe_time + dist_end[cafe_location]
                cafe_to_visit = cafe_location

        path = []

        # Getting the path from start to cafe
        prev_location = pred_start[cafe_to_visit]
        while prev_location is not None:
            path.append(prev_location)
            prev_location = pred_start[prev_location]

        path.reverse()      # Reverse the path
        path.append(cafe_to_visit)

        # Getting the path from cafe to end
        prev_location = pred_end[cafe_to_visit]
        while prev_location is not None:
            path.append(prev_location)
            prev_location = pred_end[prev_location]

        print(path)


def dijkstra(graph, start):
    """

    """
    num_locations = len(graph)

    # Initializing distance array to infinity
    dist = [float('inf')] * num_locations       # O(V)

    # Initializing predecessor array
    pred = [None] * num_locations               # O(V)

    # Set distance of the starting location to 0
    dist[start] = 0

    # Initialize priority queue
    queue = PriorityQueue()
    queue.put((0, start))

    # Main loop
    while not queue.empty():                    # O(V)
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


roads = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2),
         (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2),
         (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
cafes = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]

mygraph = RoadGraph(roads, cafes)
mygraph.routing(3, 4)


# Part 2
def optimalRoute(downhillScores, start, finish):
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

    TODO: Implement optimal route calculation
    """
