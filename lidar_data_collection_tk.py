import csv
import time
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class JackalDataCollector:
    """
    A simple GUI for collecting lidar data from a Jackal robot.
    The GUI displays the latest lidar scan data and allows the user to move the robot.
    """
    def __init__(self, master):
        self.master = master
        master.title("Jackal Lidar Data Collector")
        self.movement_after_id = None

        # Initialize the ROS node
        rospy.init_node('jackal_data_collector', anonymous=True)

        # Publisher for movement commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Initialize variable for lidar data
        self.scan_data = None
        # Subscriber for lidar scan data (assumed on topic "/scan")
        self.sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.loc_sub = rospy.Subscriber("/vrpn_client_node/jackal/pose", PoseStamped, self.pose_callback)
        # Create a matplotlib figure for plotting lidar data
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Movement control buttons
        control_frame = tk.Frame(master)
        control_frame.pack(side=tk.TOP, pady=5)

        tk.Button(control_frame, text="Forward").grid(row=0, column=1)
        tk.Button(control_frame, text="Left").grid(row=1, column=0)
        tk.Button(control_frame, text="Stop").grid(row=1, column=1)
        tk.Button(control_frame, text="Right").grid(row=1, column=2)
        tk.Button(control_frame, text="Backward").grid(row=2, column=1)

        # Bind button press and release events
        control_frame.bind_all('<ButtonPress-1>', self.on_button_press)
        control_frame.bind_all('<ButtonRelease-1>', self.on_button_release)

        # Button for saving the current lidar data
        tk.Button(master, text="Save Lidar Data", command=self.save_lidar_data).pack(side=tk.TOP, pady=5)

        # Start the periodic update for the plot
        self.update_plot()

    def on_button_press(self, event):
        widget = event.widget
        text = widget.cget('text')
        if text in ['Forward', 'Backward', 'Left', 'Right']:
            # Start repeating movement for the pressed button
            self.start_movement(text)
        elif text == 'Stop':
            self.stop()

    def on_button_release(self, event):
        # Stop repeating movement on release
        self.cancel_movement()
        self.stop()

    def start_movement(self, direction):
        """Starts sending commands continuously for the given direction."""
        self._move(direction)

    def _move(self, direction):
        """Helper method that publishes the movement command and schedules itself."""
        if direction == "Forward":
            self.move_forward()
        elif direction == "Backward":
            self.move_backward()
        elif direction == "Left":
            self.turn_left()
        elif direction == "Right":
            self.turn_right()
        # Schedule the next command in 100 ms
        self.movement_after_id = self.master.after(100, lambda: self._move(direction))

    def cancel_movement(self):
        """Cancel the scheduled movement commands."""
        if self.movement_after_id is not None:
            self.master.after_cancel(self.movement_after_id)
            self.movement_after_id = None

    def lidar_callback(self, msg):
        """Callback to store the latest lidar data."""
        self.scan_data = msg

    def pose_callback(self, msg):
        """Callback to store the latest position data."""
        self.pose = msg

    def update_plot(self):
        """Update the matplotlib plot with the latest lidar data."""
        if self.scan_data is not None:
            self.ax.clear()
            angles = np.linspace(self.scan_data.angle_min, self.scan_data.angle_max, len(self.scan_data.ranges))
            ranges = np.array(self.scan_data.ranges)
            valid = np.logical_and(ranges > self.scan_data.range_min, np.isfinite(ranges))
            print('range min:', self.scan_data.range_min)
            print('range max:', self.scan_data.range_max)
            print(len(ranges), sum(valid))
            # x = ranges * np.cos(angles)
            # y = ranges * np.sin(angles)
            x = ranges[valid] * np.cos(angles[valid])
            y = ranges[valid] * np.sin(angles[valid])
            self.ax.plot(x, y, '.', markersize=2)
            self.ax.set_title("Lidar Data")
            self.ax.set_xlabel("X (m)")
            self.ax.set_ylabel("Y (m)")
            self.ax.axis('equal')
            self.canvas.draw()
        self.master.after(1, self.update_plot)

    # Movement command functions
    def move_forward(self):
        twist = Twist()
        twist.linear.x = 0.2
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def move_backward(self):
        twist = Twist()
        twist.linear.x = -0.2
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def turn_left(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.5
        self.cmd_vel_pub.publish(twist)

    def turn_right(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = -0.5
        self.cmd_vel_pub.publish(twist)

    def stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def save_lidar_data(self):
        """Save the current lidar scan to a CSV file."""
        if self.scan_data is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = "lidardata/lidar_data_{}.csv".format(timestamp)
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write pose information if available
                if hasattr(self, 'pose') and self.pose is not None:
                    position = self.pose.pose.position
                    orientation = self.pose.pose.orientation
                    writer.writerow(["position", position.x, position.y, position.z])
                    writer.writerow(["orientation", orientation.x, orientation.y, orientation.z, orientation.w])
                else:
                    writer.writerow(["position", "N/A"])
                    writer.writerow(["orientation", "N/A"])
                writer.writerow(["angle", "range"])
                angles = np.linspace(self.scan_data.angle_min, self.scan_data.angle_max, len(self.scan_data.ranges))
                for angle, rng in zip(angles, self.scan_data.ranges):
                    writer.writerow([angle, rng])
            rospy.loginfo("Saved lidar data to %s", filename)
        else:
            rospy.logwarn("No lidar data available to save.")

if __name__ == '__main__':
    app = None
    try:
        root = tk.Tk()
        app = JackalDataCollector(root)
        root.mainloop()
    except rospy.ROSInterruptException:
        if app is not None:
            app.cancel_movement()
            app.stop()
            app.sub.unregister()
            app.cmd_vel_pub.unregister()