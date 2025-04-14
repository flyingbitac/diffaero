; Auto-generated. Do not edit!


(cl:in-package accel_control-srv)


;//! \htmlinclude DepthImage-request.msg.html

(cl:defclass <DepthImage-request> (roslisp-msg-protocol:ros-message)
  ((downsample
    :reader downsample
    :initarg :downsample
    :type cl:boolean
    :initform cl:nil)
   (post_process
    :reader post_process
    :initarg :post_process
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass DepthImage-request (<DepthImage-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DepthImage-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DepthImage-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name accel_control-srv:<DepthImage-request> is deprecated: use accel_control-srv:DepthImage-request instead.")))

(cl:ensure-generic-function 'downsample-val :lambda-list '(m))
(cl:defmethod downsample-val ((m <DepthImage-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader accel_control-srv:downsample-val is deprecated.  Use accel_control-srv:downsample instead.")
  (downsample m))

(cl:ensure-generic-function 'post_process-val :lambda-list '(m))
(cl:defmethod post_process-val ((m <DepthImage-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader accel_control-srv:post_process-val is deprecated.  Use accel_control-srv:post_process instead.")
  (post_process m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DepthImage-request>) ostream)
  "Serializes a message object of type '<DepthImage-request>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'downsample) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'post_process) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DepthImage-request>) istream)
  "Deserializes a message object of type '<DepthImage-request>"
    (cl:setf (cl:slot-value msg 'downsample) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'post_process) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DepthImage-request>)))
  "Returns string type for a service object of type '<DepthImage-request>"
  "accel_control/DepthImageRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DepthImage-request)))
  "Returns string type for a service object of type 'DepthImage-request"
  "accel_control/DepthImageRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DepthImage-request>)))
  "Returns md5sum for a message object of type '<DepthImage-request>"
  "c62de067401b31d4e5f3582933f57d5b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DepthImage-request)))
  "Returns md5sum for a message object of type 'DepthImage-request"
  "c62de067401b31d4e5f3582933f57d5b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DepthImage-request>)))
  "Returns full string definition for message of type '<DepthImage-request>"
  (cl:format cl:nil "bool downsample~%bool post_process~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DepthImage-request)))
  "Returns full string definition for message of type 'DepthImage-request"
  (cl:format cl:nil "bool downsample~%bool post_process~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DepthImage-request>))
  (cl:+ 0
     1
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DepthImage-request>))
  "Converts a ROS message object to a list"
  (cl:list 'DepthImage-request
    (cl:cons ':downsample (downsample msg))
    (cl:cons ':post_process (post_process msg))
))
;//! \htmlinclude DepthImage-response.msg.html

(cl:defclass <DepthImage-response> (roslisp-msg-protocol:ros-message)
  ((img
    :reader img
    :initarg :img
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image)))
)

(cl:defclass DepthImage-response (<DepthImage-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DepthImage-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DepthImage-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name accel_control-srv:<DepthImage-response> is deprecated: use accel_control-srv:DepthImage-response instead.")))

(cl:ensure-generic-function 'img-val :lambda-list '(m))
(cl:defmethod img-val ((m <DepthImage-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader accel_control-srv:img-val is deprecated.  Use accel_control-srv:img instead.")
  (img m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DepthImage-response>) ostream)
  "Serializes a message object of type '<DepthImage-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'img) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DepthImage-response>) istream)
  "Deserializes a message object of type '<DepthImage-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'img) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DepthImage-response>)))
  "Returns string type for a service object of type '<DepthImage-response>"
  "accel_control/DepthImageResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DepthImage-response)))
  "Returns string type for a service object of type 'DepthImage-response"
  "accel_control/DepthImageResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DepthImage-response>)))
  "Returns md5sum for a message object of type '<DepthImage-response>"
  "c62de067401b31d4e5f3582933f57d5b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DepthImage-response)))
  "Returns md5sum for a message object of type 'DepthImage-response"
  "c62de067401b31d4e5f3582933f57d5b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DepthImage-response>)))
  "Returns full string definition for message of type '<DepthImage-response>"
  (cl:format cl:nil "sensor_msgs/Image img~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DepthImage-response)))
  "Returns full string definition for message of type 'DepthImage-response"
  (cl:format cl:nil "sensor_msgs/Image img~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DepthImage-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'img))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DepthImage-response>))
  "Converts a ROS message object to a list"
  (cl:list 'DepthImage-response
    (cl:cons ':img (img msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'DepthImage)))
  'DepthImage-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'DepthImage)))
  'DepthImage-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DepthImage)))
  "Returns string type for a service object of type '<DepthImage>"
  "accel_control/DepthImage")