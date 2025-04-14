;; Auto-generated. Do not edit!


(when (boundp 'accel_control::DepthImage)
  (if (not (find-package "ACCEL_CONTROL"))
    (make-package "ACCEL_CONTROL"))
  (shadow 'DepthImage (find-package "ACCEL_CONTROL")))
(unless (find-package "ACCEL_CONTROL::DEPTHIMAGE")
  (make-package "ACCEL_CONTROL::DEPTHIMAGE"))
(unless (find-package "ACCEL_CONTROL::DEPTHIMAGEREQUEST")
  (make-package "ACCEL_CONTROL::DEPTHIMAGEREQUEST"))
(unless (find-package "ACCEL_CONTROL::DEPTHIMAGERESPONSE")
  (make-package "ACCEL_CONTROL::DEPTHIMAGERESPONSE"))

(in-package "ROS")



(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))


(defclass accel_control::DepthImageRequest
  :super ros::object
  :slots (_downsample _post_process ))

(defmethod accel_control::DepthImageRequest
  (:init
   (&key
    ((:downsample __downsample) nil)
    ((:post_process __post_process) nil)
    )
   (send-super :init)
   (setq _downsample __downsample)
   (setq _post_process __post_process)
   self)
  (:downsample
   (&optional (__downsample :null))
   (if (not (eq __downsample :null)) (setq _downsample __downsample)) _downsample)
  (:post_process
   (&optional (__post_process :null))
   (if (not (eq __post_process :null)) (setq _post_process __post_process)) _post_process)
  (:serialization-length
   ()
   (+
    ;; bool _downsample
    1
    ;; bool _post_process
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _downsample
       (if _downsample (write-byte -1 s) (write-byte 0 s))
     ;; bool _post_process
       (if _post_process (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _downsample
     (setq _downsample (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _post_process
     (setq _post_process (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass accel_control::DepthImageResponse
  :super ros::object
  :slots (_img ))

(defmethod accel_control::DepthImageResponse
  (:init
   (&key
    ((:img __img) (instance sensor_msgs::Image :init))
    )
   (send-super :init)
   (setq _img __img)
   self)
  (:img
   (&rest __img)
   (if (keywordp (car __img))
       (send* _img __img)
     (progn
       (if __img (setq _img (car __img)))
       _img)))
  (:serialization-length
   ()
   (+
    ;; sensor_msgs/Image _img
    (send _img :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; sensor_msgs/Image _img
       (send _img :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; sensor_msgs/Image _img
     (send _img :deserialize buf ptr-) (incf ptr- (send _img :serialization-length))
   ;;
   self)
  )

(defclass accel_control::DepthImage
  :super ros::object
  :slots ())

(setf (get accel_control::DepthImage :md5sum-) "c62de067401b31d4e5f3582933f57d5b")
(setf (get accel_control::DepthImage :datatype-) "accel_control/DepthImage")
(setf (get accel_control::DepthImage :request) accel_control::DepthImageRequest)
(setf (get accel_control::DepthImage :response) accel_control::DepthImageResponse)

(defmethod accel_control::DepthImageRequest
  (:response () (instance accel_control::DepthImageResponse :init)))

(setf (get accel_control::DepthImageRequest :md5sum-) "c62de067401b31d4e5f3582933f57d5b")
(setf (get accel_control::DepthImageRequest :datatype-) "accel_control/DepthImageRequest")
(setf (get accel_control::DepthImageRequest :definition-)
      "bool downsample
bool post_process
---
sensor_msgs/Image img

================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

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

(setf (get accel_control::DepthImageResponse :md5sum-) "c62de067401b31d4e5f3582933f57d5b")
(setf (get accel_control::DepthImageResponse :datatype-) "accel_control/DepthImageResponse")
(setf (get accel_control::DepthImageResponse :definition-)
      "bool downsample
bool post_process
---
sensor_msgs/Image img

================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

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



(provide :accel_control/DepthImage "c62de067401b31d4e5f3582933f57d5b")


