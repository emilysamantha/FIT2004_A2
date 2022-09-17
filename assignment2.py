"""
FIT2004 Assignment 2
Author: Emily Samantha Zarry
ID: 32558945

"""
import heapq
from collections import deque

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

        :Time Complexity: O(V + E)
        :Aux Space Complexity: O(V + E)
        """
        self.cafes = cafes                                          # O(V) space

        # Determining the number of locations (number of vertices)
        self.num_locations = get_num_vertices(roads)                # O(E)

        # Initializing adjacency list for the road network
        self.graph = self.generate_adjacency_list(roads, cafes)     # O(E)

        # Initializing adjacency list of reversed roads
        self.graph_reversed = self.generate_adjacency_list(roads, cafes, True)      # O(E)

    def routing(self, start: int, end: int) -> list:
        """
        Calculates the optimal routes for commuting while grabbing coffee along the way.

        :Input:
            start: starting location of the journey
            end: ending location of the journey

        :Output: the shortest route from the start location to the end location,
                 going through at least 1 of the locations listed in cafes,
                 if no such route exists, returns None

        :Time Complexity: (E log V)
        :Aux Space Complexity: O(V + E)
        """
        # Apply Dijkstra's algorithm with the start location as source
        dist_start, pred_start = dijkstra(self.graph, start)            # O(E log V)

        # Apply Dijkstra's algorithm with the ending location as source
        # We pass the graph with the roads reversed to maintain the reachability
        dist_end, pred_end = dijkstra(self.graph_reversed, end)         # O(E log V)

        # Get the quickest time from start to a cafe and from that cafe to the end
        min_time, cafe_to_visit = self.get_min(dist_start, dist_end)    # O(V), E dominates V since graph is connected

        # If we found no cafes to visit, then no possible routes exist, return None
        if cafe_to_visit is None:
            return None

        return self.construct_path(cafe_to_visit, pred_start, pred_end)     # O(V)

    def generate_adjacency_list(self, roads: list, cafes: list, reverse=False) -> list:
        """
        Method to generate an adjacency list given the list of roads and cafes.

        :Input:
            roads: list of roads represented as tuples (u, v, w), where
                    u is the starting location ID for a road,
                    v is the ending location ID for a road,
                    w is the time taken to travel from a location u to a location v
            cafes: list of cafes represented as tuples (location, waiting_time), where
                    location is the location of the cafe,
                    waiting_time is the waiting time for a coffee in the cafe
            reverse: boolean value to indicate whether the road direction should be reversed

        :Output:
            graph: adjacency list of the given values

        :Time Complexity: O(E)
        :Aux Space Complexity: O(V+E)
        """
        # Initializing the graph
        # The first element of the inner list is reserved to indicate if there is a cafe in that location
        # If there is no cafe, the first element is None
        # If there is a cafe, the first element stores the waiting time for that cafe
        graph = [[None] for _ in range(self.num_locations + 1)]

        # Adding cafe locations and the waiting time
        for i in range(len(cafes)):                                 # O(V)
            cafe_location, waiting_time = cafes[i]
            graph[cafe_location][0] = waiting_time                  # Setting the waiting time for that cafe

        # Filling the adjacency list
        for i in range(len(roads)):                                 # O(E)
            start_location, end_location, travel_time = roads[i]
            if not reverse:
                graph[start_location].append((end_location, travel_time))
            else:
                graph[end_location].append((start_location, travel_time))

        return graph

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
        for cafe_location, cafe_time in self.cafes:             # O(V)
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
        path = []           # O(V) space

        # Getting the path from start to cafe
        prev_location = pred_start[cafe_to_visit]
        while prev_location is not None:                # O(V)
            path.append(prev_location)
            prev_location = pred_start[prev_location]

        path.reverse()  # Reverse the path              # O(V)
        path.append(cafe_to_visit)

        # Getting the path from cafe to end
        prev_location = pred_end[cafe_to_visit]
        while prev_location is not None:                # O(V)
            path.append(prev_location)
            prev_location = pred_end[prev_location]

        return path


def get_num_vertices(data: list) -> int:
    """
    Method to get the number of vertices in the data by getting the largest number.

    :Input:
        data: list of data that represents the connections in a graph
    :Output:
        num_vertices: the number of vertices in the given data

    :Time Complexity: O(E)
    :Aux Space Complexity: O(1)
    """
    num_vertices = 0

    for i in range(len(data)):          # O(E)
        if data[i][0] > num_vertices:
            num_vertices = data[i][0]
        if data[i][1] > num_vertices:
            num_vertices = data[i][1]

    return num_vertices


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
    dist = [float('inf')] * num_locations               # O(V) space

    # Initializing predecessor array
    pred = [None] * num_locations                       # O(V) space

    # Setting distance of the starting location to 0
    if len(graph) >= start:
        dist[start] = 0

    # Initializing priority queue
    queue = []                                          # O(V) space
    heapq.heapify(queue)
    heapq.heappush(queue, (0, start))                   # O(E log V)

    # Main loop
    while len(queue) > 0:                               # O(V)
        # Extracting the nearest location u
        dist_u, u = heapq.heappop(queue)                # O(1)

        # Only processing up-to-date entries
        if dist_u == dist[u]:
            # For each neighbour v of u, relax along that edge
            for i in range(1, len(graph[u])):  # Skipping the first element (Waiting time)
                v, time = graph[u][i]

                # If distance of u plus travel time of that road is quicker than the current distance of v
                if dist_u + time < dist[v]:
                    # Update the minimum distance of v
                    dist[v] = dist[u] + time
                    # Set the predecessor of v as u
                    pred[v] = u
                    # Update the priority queue
                    heapq.heappush(queue, (dist[v], v))

    return dist, pred


# Part 2
def optimalRoute(downhillScores: list, start: int, finish: int) -> list:
    """
    Calculates the route that should be used for going from the starting point to the
    finishing point while using only downhill segments and obtaining the maximum score.

    :Input:
        downhillScores:
            list of downhill segments represented as tuples (a, b, c), where
            a is the start point of a downhill segment,
            b is the end point of a downhill segment
            c is the integer score for using this downhill segment to go from point a to b
        start: the start point of the tournament
        finish: the end point of the tournament

    :Output: the optimal route for obtaining the maximum score

    :Time Complexity: O(D), where D is the number of downhill segments (edges)
    :Aux Space Complexity: O(D), where D is the number of downhill segments (edges)

    Approach Explanation:
    To prevent iterating the edge relaxation V times, a queue is used to keep track of the order of vertices
    we need to visit. This queue stores the vertices such that we process them in the order of its reachability.
    This way, for each vertex we visit, it is guaranteed that we already know the maximum distance of all its
    predecessor vertices. Hence, there is no need to re-iterate V times, and we can process each edge only once,
    giving a time complexity of O(D).

    So for each vertex, beginning from the starting vertex, we append the vertex's next reachable vertex to the queue
    and this next vertex/vertices will be the next vertex we will process in our loop.

    This gives a total upper bound of O(D) for the whole loop.
    """
    # If the start and finish points are the same, return that point as the path
    if start == finish:
        return [start]

    # Getting the number of intersections
    num_intersections = get_num_vertices(downhillScores)        # O(D)

    # Initializing adjacency list, O(D)
    graph_next = construct_adjacency_list(downhillScores, num_intersections)            # keeps track of outgoing edges
    graph_before = construct_adjacency_list(downhillScores, num_intersections, True)    # keeps track of incoming edges

    # Initializing distance array to infinity
    dist = [-float('inf')] * (num_intersections + 1)            # O(P) space

    # Initializing predecessor array
    pred = [None] * (num_intersections + 1)                     # O(P) space

    # Setting distance of the starting location to 0
    dist[start] = 0

    # Initializing array to detect whether an intersection has been included
    included = [False] * (num_intersections + 1)                # O(P) space

    # Initializing queue to keep track of the order of vertices to visit
    vertices_to_visit = deque()

    vertices_to_visit.append(start)                             # O(1)

    # While there are still vertices to uncover
    while len(vertices_to_visit) > 0:                           # Total O(D)
        # Get the current vertex we are looking at
        curr_vertex = vertices_to_visit.popleft()               # O(1)

        # For each vertex, go through each of the incoming edges
        for i in range(len(graph_before[curr_vertex])):
            # Get the previous vertex and the score for going through that downhill segment
            prev, score = graph_before[curr_vertex][i]

            # If the score for going through that downhill segment is larger than the current score
            if dist[prev] + score > dist[curr_vertex]:
                # Update the maximum score of curr_vertex
                dist[curr_vertex] = dist[prev] + score
                # Set the predecessor of curr_vertex as the previous vertex
                pred[curr_vertex] = prev

        # If there are outgoing edges from the current vertex
        if len(graph_next[curr_vertex]) > 0:
            # Go through each edge
            for i in range(len(graph_next[curr_vertex])):
                # If the vertex of that outgoing edge has not been included in the queue
                if not included[graph_next[curr_vertex][i][0]]:
                    # Append the vertex to vertices_to_visit
                    vertices_to_visit.append(graph_next[curr_vertex][i][0])
                    # Mark the vertex as included
                    included[graph_next[curr_vertex][i][0]] = True

    return construct_path(pred, finish)                         # O(P)


def construct_adjacency_list(downhillScores: list, num_intersections: int, before=False) -> list:
    """
    Method to construct adjacency list given the list of downhill scores.

    :Input:
        downhillScores:
            list of downhill segments represented as tuples (a, b, c), where
            a is the start point of a downhill segment,
            b is the end point of a downhill segment
            c is the integer score for using this downhill segment to go from point a to b
        num_intersections: the total number of intersections in the tournament
        before:
            boolean to indicate whether to store the incoming neighbor.
            true if we want to store the outgoing neighbor,
            false if we want to store the incoming neighbor.
    :Output: adjacency list of the given values

    :Time Complexity: O(D)
    :Aux Space Complexity: O(D)
    """
    # Initializing adjacency list
    graph = [[] for _ in range(num_intersections + 1)]
    # O(P + D) space, but D dominates P
    # So, space complexity becomes O(D)

    # Filling the adjacency list from the data in downhillScores
    for i in range(len(downhillScores)):                        # O(D)
        start_vertex, end_vertex, score = downhillScores[i]

        if not before:
            graph[start_vertex].append((end_vertex, score))
        else:
            graph[end_vertex].append((start_vertex, score))

    return graph


def construct_path(pred: list, finish: int) -> list:
    """
    Method to construct the path from start to finish that gives the maximum score.

    :Input:
        pred: predecessor array
        finish: the end point of the tournament
    :Output:
        path: list of intersection points to visit to obtain the maximum score

    :Time Complexity: O(P)
    :Aux Space Complexity: O(P)
    """
    # If the end point of the tournament is not reachable, return an empty list
    if pred[finish] is None:
        return None

    # Constructing the optimal route for obtaining the maximum score
    path = [finish]
    prev_intersection = pred[finish]
    while prev_intersection is not None:
        path.append(prev_intersection)
        prev_intersection = pred[prev_intersection]
    path.reverse()

    return path



