; Auto-generated. Do not edit!


(cl:in-package realsense2_camera-srv)


;//! \htmlinclude DeviceInfo-request.msg.html

(cl:defclass <DeviceInfo-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass DeviceInfo-request (<DeviceInfo-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DeviceInfo-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DeviceInfo-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name realsense2_camera-srv:<DeviceInfo-request> is deprecated: use realsense2_camera-srv:DeviceInfo-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DeviceInfo-request>) ostream)
  "Serializes a message object of type '<DeviceInfo-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DeviceInfo-request>) istream)
  "Deserializes a message object of type '<DeviceInfo-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DeviceInfo-request>)))
  "Returns string type for a service object of type '<DeviceInfo-request>"
  "realsense2_camera/DeviceInfoRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DeviceInfo-request)))
  "Returns string type for a service object of type 'DeviceInfo-request"
  "realsense2_camera/DeviceInfoRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DeviceInfo-request>)))
  "Returns md5sum for a message object of type '<DeviceInfo-request>"
  "914e9cfa74a4f66f08c3fe1016943c1b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DeviceInfo-request)))
  "Returns md5sum for a message object of type 'DeviceInfo-request"
  "914e9cfa74a4f66f08c3fe1016943c1b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DeviceInfo-request>)))
  "Returns full string definition for message of type '<DeviceInfo-request>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DeviceInfo-request)))
  "Returns full string definition for message of type 'DeviceInfo-request"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DeviceInfo-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DeviceInfo-request>))
  "Converts a ROS message object to a list"
  (cl:list 'DeviceInfo-request
))
;//! \htmlinclude DeviceInfo-response.msg.html

(cl:defclass <DeviceInfo-response> (roslisp-msg-protocol:ros-message)
  ((device_name
    :reader device_name
    :initarg :device_name
    :type cl:string
    :initform "")
   (serial_number
    :reader serial_number
    :initarg :serial_number
    :type cl:string
    :initform "")
   (firmware_version
    :reader firmware_version
    :initarg :firmware_version
    :type cl:string
    :initform "")
   (usb_type_descriptor
    :reader usb_type_descriptor
    :initarg :usb_type_descriptor
    :type cl:string
    :initform "")
   (firmware_update_id
    :reader firmware_update_id
    :initarg :firmware_update_id
    :type cl:string
    :initform "")
   (sensors
    :reader sensors
    :initarg :sensors
    :type cl:string
    :initform ""))
)

(cl:defclass DeviceInfo-response (<DeviceInfo-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DeviceInfo-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DeviceInfo-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name realsense2_camera-srv:<DeviceInfo-response> is deprecated: use realsense2_camera-srv:DeviceInfo-response instead.")))

(cl:ensure-generic-function 'device_name-val :lambda-list '(m))
(cl:defmethod device_name-val ((m <DeviceInfo-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-srv:device_name-val is deprecated.  Use realsense2_camera-srv:device_name instead.")
  (device_name m))

(cl:ensure-generic-function 'serial_number-val :lambda-list '(m))
(cl:defmethod serial_number-val ((m <DeviceInfo-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-srv:serial_number-val is deprecated.  Use realsense2_camera-srv:serial_number instead.")
  (serial_number m))

(cl:ensure-generic-function 'firmware_version-val :lambda-list '(m))
(cl:defmethod firmware_version-val ((m <DeviceInfo-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-srv:firmware_version-val is deprecated.  Use realsense2_camera-srv:firmware_version instead.")
  (firmware_version m))

(cl:ensure-generic-function 'usb_type_descriptor-val :lambda-list '(m))
(cl:defmethod usb_type_descriptor-val ((m <DeviceInfo-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-srv:usb_type_descriptor-val is deprecated.  Use realsense2_camera-srv:usb_type_descriptor instead.")
  (usb_type_descriptor m))

(cl:ensure-generic-function 'firmware_update_id-val :lambda-list '(m))
(cl:defmethod firmware_update_id-val ((m <DeviceInfo-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-srv:firmware_update_id-val is deprecated.  Use realsense2_camera-srv:firmware_update_id instead.")
  (firmware_update_id m))

(cl:ensure-generic-function 'sensors-val :lambda-list '(m))
(cl:defmethod sensors-val ((m <DeviceInfo-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader realsense2_camera-srv:sensors-val is deprecated.  Use realsense2_camera-srv:sensors instead.")
  (sensors m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DeviceInfo-response>) ostream)
  "Serializes a message object of type '<DeviceInfo-response>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'device_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'device_name))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'serial_number))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'serial_number))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'firmware_version))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'firmware_version))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'usb_type_descriptor))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'usb_type_descriptor))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'firmware_update_id))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'firmware_update_id))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'sensors))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'sensors))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DeviceInfo-response>) istream)
  "Deserializes a message object of type '<DeviceInfo-response>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'device_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'device_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'serial_number) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'serial_number) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'firmware_version) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'firmware_version) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'usb_type_descriptor) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'usb_type_descriptor) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'firmware_update_id) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'firmware_update_id) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'sensors) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'sensors) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DeviceInfo-response>)))
  "Returns string type for a service object of type '<DeviceInfo-response>"
  "realsense2_camera/DeviceInfoResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DeviceInfo-response)))
  "Returns string type for a service object of type 'DeviceInfo-response"
  "realsense2_camera/DeviceInfoResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DeviceInfo-response>)))
  "Returns md5sum for a message object of type '<DeviceInfo-response>"
  "914e9cfa74a4f66f08c3fe1016943c1b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DeviceInfo-response)))
  "Returns md5sum for a message object of type 'DeviceInfo-response"
  "914e9cfa74a4f66f08c3fe1016943c1b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DeviceInfo-response>)))
  "Returns full string definition for message of type '<DeviceInfo-response>"
  (cl:format cl:nil "string device_name~%string serial_number~%string firmware_version~%string usb_type_descriptor~%string firmware_update_id~%string sensors~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DeviceInfo-response)))
  "Returns full string definition for message of type 'DeviceInfo-response"
  (cl:format cl:nil "string device_name~%string serial_number~%string firmware_version~%string usb_type_descriptor~%string firmware_update_id~%string sensors~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DeviceInfo-response>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'device_name))
     4 (cl:length (cl:slot-value msg 'serial_number))
     4 (cl:length (cl:slot-value msg 'firmware_version))
     4 (cl:length (cl:slot-value msg 'usb_type_descriptor))
     4 (cl:length (cl:slot-value msg 'firmware_update_id))
     4 (cl:length (cl:slot-value msg 'sensors))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DeviceInfo-response>))
  "Converts a ROS message object to a list"
  (cl:list 'DeviceInfo-response
    (cl:cons ':device_name (device_name msg))
    (cl:cons ':serial_number (serial_number msg))
    (cl:cons ':firmware_version (firmware_version msg))
    (cl:cons ':usb_type_descriptor (usb_type_descriptor msg))
    (cl:cons ':firmware_update_id (firmware_update_id msg))
    (cl:cons ':sensors (sensors msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'DeviceInfo)))
  'DeviceInfo-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'DeviceInfo)))
  'DeviceInfo-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DeviceInfo)))
  "Returns string type for a service object of type '<DeviceInfo>"
  "realsense2_camera/DeviceInfo")