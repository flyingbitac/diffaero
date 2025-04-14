;; Auto-generated. Do not edit!


(when (boundp 'realsense2_camera::Extrinsics)
  (if (not (find-package "REALSENSE2_CAMERA"))
    (make-package "REALSENSE2_CAMERA"))
  (shadow 'Extrinsics (find-package "REALSENSE2_CAMERA")))
(unless (find-package "REALSENSE2_CAMERA::EXTRINSICS")
  (make-package "REALSENSE2_CAMERA::EXTRINSICS"))

(in-package "ROS")
;;//! \htmlinclude Extrinsics.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass realsense2_camera::Extrinsics
  :super ros::object
  :slots (_header _rotation _translation ))

(defmethod realsense2_camera::Extrinsics
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:rotation __rotation) (make-array 9 :initial-element 0.0 :element-type :float))
    ((:translation __translation) (make-array 3 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _header __header)
   (setq _rotation __rotation)
   (setq _translation __translation)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:rotation
   (&optional __rotation)
   (if __rotation (setq _rotation __rotation)) _rotation)
  (:translation
   (&optional __translation)
   (if __translation (setq _translation __translation)) _translation)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float64[9] _rotation
    (* 8    9)
    ;; float64[3] _translation
    (* 8    3)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float64[9] _rotation
     (dotimes (i 9)
       (sys::poke (elt _rotation i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[3] _translation
     (dotimes (i 3)
       (sys::poke (elt _translation i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float64[9] _rotation
   (dotimes (i (length _rotation))
     (setf (elt _rotation i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[3] _translation
   (dotimes (i (length _translation))
     (setf (elt _translation i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;;
   self)
  )

(setf (get realsense2_camera::Extrinsics :md5sum-) "3627b43073f4cd5dd6dc179a49eda2ad")
(setf (get realsense2_camera::Extrinsics :datatype-) "realsense2_camera/Extrinsics")
(setf (get realsense2_camera::Extrinsics :definition-)
      "std_msgs/Header header
float64[9] rotation
float64[3] translation

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

")



(provide :realsense2_camera/Extrinsics "3627b43073f4cd5dd6dc179a49eda2ad")


