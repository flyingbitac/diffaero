; Auto-generated. Do not edit!


(cl:in-package realsense2_camera-msg)


;//! \htmlinclude Extrinsics.msg.html

(cl:defclass <Extrinsics> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (rotation
    :reader rotation
    :initarg :rotation
    :type (cl:vector cl:float)
   :initform (cl:make-array 9 :element-type 'cl:float :initial-element 0.0))
   (translation
    :reader translation
    :initarg :translation
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass Extrinsics (<Extrinsics>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Extrinsics>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Extrinsics)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name realsense2_camera-msg:<Extrinsics> is deprecated: use realsense2_camera-msg:Extrinsics instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <Extrinsics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-msg:header-val is deprecated.  Use realsense2_camera-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'rotation-val :lambda-list '(m))
(cl:defmethod rotation-val ((m <Extrinsics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-msg:rotation-val is deprecated.  Use realsense2_camera-msg:rotation instead.")
  (rotation m))

(cl:ensure-generic-function 'translation-val :lambda-list '(m))
(cl:defmethod translation-val ((m <Extrinsics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-msg:translation-val is deprecated.  Use realsense2_camera-msg:translation instead.")
  (translation m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Extrinsics>) ostream)
  "Serializes a message object of type '<Extrinsics>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'rotation))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'translation))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Extrinsics>) istream)
  "Deserializes a message object of type '<Extrinsics>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:setf (cl:slot-value msg 'rotation) (cl:make-array 9))
  (cl:let ((vals (cl:slot-value msg 'rotation)))
    (cl:dotimes (i 9)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'translation) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'translation)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Extrinsics>)))
  "Returns string type for a message object of type '<Extrinsics>"
  "realsense2_camera/Extrinsics")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Extrinsics)))
  "Returns string type for a message object of type 'Extrinsics"
  "realsense2_camera/Extrinsics")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Extrinsics>)))
  "Returns md5sum for a message object of type '<Extrinsics>"
  "3627b43073f4cd5dd6dc179a49eda2ad")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Extrinsics)))
  "Returns md5sum for a message object of type 'Extrinsics"
  "3627b43073f4cd5dd6dc179a49eda2ad")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Extrinsics>)))
  "Returns full string definition for message of type '<Extrinsics>"
  (cl:format cl:nil "std_msgs/Header header~%float64[9] rotation~%float64[3] translation~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Extrinsics)))
  "Returns full string definition for message of type 'Extrinsics"
  (cl:format cl:nil "std_msgs/Header header~%float64[9] rotation~%float64[3] translation~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Extrinsics>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'rotation) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'translation) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Extrinsics>))
  "Converts a ROS message object to a list"
  (cl:list 'Extrinsics
    (cl:cons ':header (header msg))
    (cl:cons ':rotation (rotation msg))
    (cl:cons ':translation (translation msg))
))
