; Auto-generated. Do not edit!


(cl:in-package realsense2_camera-msg)


;//! \htmlinclude IMUInfo.msg.html

(cl:defclass <IMUInfo> (roslisp-msg-protocol:ros-message)
  ((frame_id
    :reader frame_id
    :initarg :frame_id
    :type cl:string
    :initform "")
   (data
    :reader data
    :initarg :data
    :type (cl:vector cl:float)
   :initform (cl:make-array 12 :element-type 'cl:float :initial-element 0.0))
   (noise_variances
    :reader noise_variances
    :initarg :noise_variances
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (bias_variances
    :reader bias_variances
    :initarg :bias_variances
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass IMUInfo (<IMUInfo>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <IMUInfo>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'IMUInfo)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name realsense2_camera-msg:<IMUInfo> is deprecated: use realsense2_camera-msg:IMUInfo instead.")))

(cl:ensure-generic-function 'frame_id-val :lambda-list '(m))
(cl:defmethod frame_id-val ((m <IMUInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-msg:frame_id-val is deprecated.  Use realsense2_camera-msg:frame_id instead.")
  (frame_id m))

(cl:ensure-generic-function 'data-val :lambda-list '(m))
(cl:defmethod data-val ((m <IMUInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-msg:data-val is deprecated.  Use realsense2_camera-msg:data instead.")
  (data m))

(cl:ensure-generic-function 'noise_variances-val :lambda-list '(m))
(cl:defmethod noise_variances-val ((m <IMUInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-msg:noise_variances-val is deprecated.  Use realsense2_camera-msg:noise_variances instead.")
  (noise_variances m))

(cl:ensure-generic-function 'bias_variances-val :lambda-list '(m))
(cl:defmethod bias_variances-val ((m <IMUInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-msg:bias_variances-val is deprecated.  Use realsense2_camera-msg:bias_variances instead.")
  (bias_variances m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <IMUInfo>) ostream)
  "Serializes a message object of type '<IMUInfo>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'frame_id))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'frame_id))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'data))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'noise_variances))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'bias_variances))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <IMUInfo>) istream)
  "Deserializes a message object of type '<IMUInfo>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'frame_id) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'frame_id) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  (cl:setf (cl:slot-value msg 'data) (cl:make-array 12))
  (cl:let ((vals (cl:slot-value msg 'data)))
    (cl:dotimes (i 12)
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
  (cl:setf (cl:slot-value msg 'noise_variances) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'noise_variances)))
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
  (cl:setf (cl:slot-value msg 'bias_variances) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'bias_variances)))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<IMUInfo>)))
  "Returns string type for a message object of type '<IMUInfo>"
  "realsense2_camera/IMUInfo")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'IMUInfo)))
  "Returns string type for a message object of type 'IMUInfo"
  "realsense2_camera/IMUInfo")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<IMUInfo>)))
  "Returns md5sum for a message object of type '<IMUInfo>"
  "a02adb3a99530b11ba18a16f40f9512a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'IMUInfo)))
  "Returns md5sum for a message object of type 'IMUInfo"
  "a02adb3a99530b11ba18a16f40f9512a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<IMUInfo>)))
  "Returns full string definition for message of type '<IMUInfo>"
  (cl:format cl:nil "string frame_id~%float64[12] data~%float64[3] noise_variances~%float64[3] bias_variances~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'IMUInfo)))
  "Returns full string definition for message of type 'IMUInfo"
  (cl:format cl:nil "string frame_id~%float64[12] data~%float64[3] noise_variances~%float64[3] bias_variances~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <IMUInfo>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'frame_id))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'data) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'noise_variances) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'bias_variances) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <IMUInfo>))
  "Converts a ROS message object to a list"
  (cl:list 'IMUInfo
    (cl:cons ':frame_id (frame_id msg))
    (cl:cons ':data (data msg))
    (cl:cons ':noise_variances (noise_variances msg))
    (cl:cons ':bias_variances (bias_variances msg))
))
