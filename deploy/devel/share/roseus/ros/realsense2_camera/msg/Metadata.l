;; Auto-generated. Do not edit!


(when (boundp 'realsense2_camera::Metadata)
  (if (not (find-package "REALSENSE2_CAMERA"))
    (make-package "REALSENSE2_CAMERA"))
  (shadow 'Metadata (find-package "REALSENSE2_CAMERA")))
(unless (find-package "REALSENSE2_CAMERA::METADATA")
  (make-package "REALSENSE2_CAMERA::METADATA"))

(in-package "ROS")
;;//! \htmlinclude Metadata.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass realsense2_camera::Metadata
  :super ros::object
  :slots (_header _json_data ))

(defmethod realsense2_camera::Metadata
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:json_data __json_data) "")
    )
   (send-super :init)
   (setq _header __header)
   (setq _json_data (string __json_data))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:json_data
   (&optional __json_data)
   (if __json_data (setq _json_data __json_data)) _json_data)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; string _json_data
    4 (length _json_data)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; string _json_data
       (write-long (length _json_data) s) (princ _json_data s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; string _json_data
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _json_data (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get realsense2_camera::Metadata :md5sum-) "4966ca002be16ee67fe4dbfb2f354787")
(setf (get realsense2_camera::Metadata :datatype-) "realsense2_camera/Metadata")
(setf (get realsense2_camera::Metadata :definition-)
      "std_msgs/Header header
string json_data
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



(provide :realsense2_camera/Metadata "4966ca002be16ee67fe4dbfb2f354787")


