# ============================================================
# tracker.py
# PURPOSE: Manages object tracking state and trail history
# This is the MEMORY of our detection system
# ============================================================

from collections import defaultdict
import numpy as np
import cv2


# ------------------------------------------------------------
# TRACK HISTORY MANAGER
# Keeps a record of where each tracked object has been
# ------------------------------------------------------------
class TrackHistory:

    def __init__(self, max_trail_length=30):
        """
        Initializes the track history manager.

        Parameters:
            max_trail_length : how many past positions to remember
                               30 = last 1 second at 30fps
        
        WHY track history?
        We can draw a TRAIL showing where an object came from.
        Like motion blur but drawn as a line. Very cool visually
        and useful for understanding movement patterns.
        """

        # defaultdict automatically creates an empty list
        # for any new track ID we haven't seen before
        # Regular dict would crash with KeyError on new IDs
        self.trails = defaultdict(list)
        self.max_trail_length = max_trail_length

        # Count how many frames each object has been tracked
        self.frame_counts = defaultdict(int)

        # Store the last known class name for each track ID
        self.class_names = {}

        # Store color for each track ID (consistent across frames)
        self.colors = {}


    def update(self, track_id, center_point, class_name, color):
        """
        Updates the history for one tracked object.

        Parameters:
            track_id     : unique integer ID for this object
            center_point : (x, y) tuple — center of bounding box
            class_name   : what type of object this is
            color        : BGR color tuple for drawing
        
        HOW center point is calculated:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
        """

        # Add new position to this object's trail
        self.trails[track_id].append(center_point)

        # Keep trail length limited to max_trail_length
        # If trail is too long, remove the oldest point
        if len(self.trails[track_id]) > self.max_trail_length:
            self.trails[track_id].pop(0)

        # Increment frame count for this object
        self.frame_counts[track_id] += 1

        # Store class name and color
        self.class_names[track_id] = class_name
        self.colors[track_id]      = color


    def get_trail(self, track_id):
        """
        Returns the list of past positions for a tracked object.
        Used for drawing the movement trail on the video.
        """
        return self.trails.get(track_id, [])


    def draw_trails(self, frame):
        """
        Draws movement trails for ALL tracked objects on the frame.
        
        The trail fades from thick+bright at current position
        to thin+dim at oldest position — like a comet tail effect.

        Parameters:
            frame : the current video frame to draw on
        
        Returns:
            frame : the same frame with trails drawn on it
        """

        for track_id, trail in self.trails.items():

            # Need at least 2 points to draw a line
            if len(trail) < 2:
                continue

            color = self.colors.get(track_id, (255, 255, 255))

            # Draw lines between consecutive trail points
            # Enumerate gives us index so we can calculate fade
            for i in range(1, len(trail)):

                # Calculate opacity based on position in trail
                # Recent points = more opaque, old points = transparent
                alpha     = i / len(trail)
                thickness = max(1, int(alpha * 3))

                # Fade the color based on alpha
                faded_color = tuple(int(c * alpha) for c in color)

                # Draw line segment between two consecutive points
                cv2.line(
                    frame,
                    trail[i - 1],   # previous point
                    trail[i],       # current point
                    faded_color,
                    thickness
                )

        return frame


    def clean_lost_tracks(self, active_ids):
        """
        Removes trail history for objects no longer being tracked.
        
        WHY: If we never clean up, memory keeps growing as new
        objects appear and disappear. Could cause memory issues
        in long-running videos.

        Parameters:
            active_ids : set of track IDs currently detected
        """

        # Find IDs in our history that are no longer active
        lost_ids = set(self.trails.keys()) - set(active_ids)

        for lost_id in lost_ids:
            # Remove from all tracking dictionaries
            self.trails.pop(lost_id, None)
            self.frame_counts.pop(lost_id, None)
            self.class_names.pop(lost_id, None)
            self.colors.pop(lost_id, None)


    def get_object_count(self):
        """
        Returns total number of objects currently being tracked.
        """
        return len(self.trails)


    def get_stats(self):
        """
        Returns tracking statistics for display in the UI.
        
        Returns a dictionary with:
        - total tracked objects
        - class breakdown (how many persons, cars etc)
        """

        stats = {
            "total_tracked" : len(self.trails),
            "class_counts"  : {}
        }

        # Count how many objects of each class are being tracked
        for track_id, class_name in self.class_names.items():
            if class_name not in stats["class_counts"]:
                stats["class_counts"][class_name] = 0
            stats["class_counts"][class_name] += 1

        return stats


# ------------------------------------------------------------
# ZONE COUNTER (Bonus Feature)
# Counts how many objects pass through a defined region
# ------------------------------------------------------------
class ZoneCounter:

    def __init__(self, zone_points):
        """
        Creates a counting zone defined by polygon points.
        
        Parameters:
            zone_points : list of (x,y) tuples defining the zone
                         Example: [(100,200), (400,200), (400,400), (100,400)]
        
        Real world use: Count how many people enter a shop,
        how many cars pass a toll booth, etc.
        """
        self.zone_points = np.array(zone_points, dtype=np.int32)
        self.counted_ids = set()  # IDs we already counted
        self.count       = 0


    def update(self, track_id, center_point):
        """
        Checks if an object's center is inside the zone.
        If yes and not counted yet, increment count.

        Parameters:
            track_id     : unique ID of the tracked object
            center_point : (x, y) center of the object

        Returns:
            True if object is inside zone, False otherwise
        """

        # cv2.pointPolygonTest checks if point is inside polygon
        # Returns positive if inside, negative if outside
        result = cv2.pointPolygonTest(
            self.zone_points,
            center_point,
            False  # False = just inside/outside, not distance
        )

        is_inside = result >= 0

        # Count this object only once even if it stays in zone
        if is_inside and track_id not in self.counted_ids:
            self.counted_ids.add(track_id)
            self.count += 1

        return is_inside


    def draw_zone(self, frame, color=(0, 255, 255)):
        """
        Draws the counting zone as a semi-transparent polygon.

        Parameters:
            frame : video frame to draw on
            color : BGR color for the zone (default = yellow)
        
        Returns:
            frame with zone drawn on it
        """

        # Create a copy for transparency effect
        overlay = frame.copy()

        # Fill the zone polygon with semi-transparent color
        cv2.fillPoly(overlay, [self.zone_points], color)

        # Blend overlay with original frame
        # alpha=0.25 means 25% zone color, 75% original frame
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Draw solid border around zone
        cv2.polylines(
            frame,
            [self.zone_points],
            isClosed=True,
            color=color,
            thickness=2
        )

        # Show count inside zone
        zone_center_x = int(np.mean(self.zone_points[:, 0]))
        zone_center_y = int(np.mean(self.zone_points[:, 1]))

        cv2.putText(
            frame,
            f"Count: {self.count}",
            (zone_center_x - 40, zone_center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

        return frame


# ------------------------------------------------------------
# TEST BLOCK
# ------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 50)
    print("TESTING tracker.py")
    print("=" * 50)

    # Create track history
    history = TrackHistory(max_trail_length=30)

    # Simulate 5 frames of a person moving right
    print("\n📌 Simulating person moving right across 5 frames...")
    for frame_num in range(5):
        x_position = 100 + (frame_num * 20)  # moves right
        center     = (x_position, 200)
        history.update(
            track_id    = 1,
            center_point= center,
            class_name  = "person",
            color       = (0, 255, 0)
        )

    trail = history.get_trail(track_id=1)
    print(f"Trail points recorded: {len(trail)}")
    print(f"Trail positions: {trail}")

    # Test stats
    stats = history.get_stats()
    print(f"\n📌 Stats: {stats}")

    # Test zone counter
    print("\n📌 Testing ZoneCounter...")
    zone  = ZoneCounter([(50, 50), (300, 50), (300, 300), (50, 300)])
    inside = zone.update(track_id=1, center_point=(150, 150))
    outside= zone.update(track_id=2, center_point=(400, 400))
    print(f"Point (150,150) inside zone: {inside} (expected True)")
    print(f"Point (400,400) inside zone: {outside} (expected False)")
    print(f"Zone count: {zone.count} (expected 1)")

    print("\n" + "=" * 50)
    print("✅ tracker.py tests complete!")
    print("=" * 50)