; Auto-generated. Do not edit!


(cl:in-package realsense2_camera-msg)


;//! \htmlinclude Metadata.msg.html

(cl:defclass <Metadata> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (json_data
    :reader json_data
    :initarg :json_data
    :type cl:string
    :initform ""))
)

(cl:defclass Metadata (<Metadata>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Metadata>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Metadata)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name realsense2_camera-msg:<Metadata> is deprecated: use realsense2_camera-msg:Metadata instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <Metadata>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-msg:header-val is deprecated.  Use realsense2_camera-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'json_data-val :lambda-list '(m))
(cl:defmethod json_data-val ((m <Metadata>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-msg:json_data-val is deprecated.  Use realsense2_camera-msg:json_data instead.")
  (json_data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Metadata>) ostream)
  "Serializes a message object of type '<Metadata>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'json_data))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'json_data))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Metadata>) istream)
  "Deserializes a message object of type '<Metadata>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'json_data) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'json_data) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Metadata>)))
  "Returns string type for a message object of type '<Metadata>"
  "realsense2_camera/Metadata")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Metadata)))
  "Returns string type for a message object of type 'Metadata"
  "realsense2_camera/Metadata")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Metadata>)))
  "Returns md5sum for a message object of type '<Metadata>"
  "4966ca002be16ee67fe4dbfb2f354787")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Metadata)))
  "Returns md5sum for a message object of type 'Metadata"
  "4966ca002be16ee67fe4dbfb2f354787")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Metadata>)))
  "Returns full string definition for message of type '<Metadata>"
  (cl:format cl:nil "std_msgs/Header header~%string json_data~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Metadata)))
  "Returns full string definition for message of type 'Metadata"
  (cl:format cl:nil "std_msgs/Header header~%string json_data~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Metadata>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:length (cl:slot-value msg 'json_data))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Metadata>))
  "Converts a ROS message object to a list"
  (cl:list 'Metadata
    (cl:cons ':header (header msg))
    (cl:cons ':json_data (json_data msg))
))
