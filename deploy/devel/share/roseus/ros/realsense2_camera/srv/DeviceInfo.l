;; Auto-generated. Do not edit!


(when (boundp 'realsense2_camera::DeviceInfo)
  (if (not (find-package "REALSENSE2_CAMERA"))
    (make-package "REALSENSE2_CAMERA"))
  (shadow 'DeviceInfo (find-package "REALSENSE2_CAMERA")))
(unless (find-package "REALSENSE2_CAMERA::DEVICEINFO")
  (make-package "REALSENSE2_CAMERA::DEVICEINFO"))
(unless (find-package "REALSENSE2_CAMERA::DEVICEINFOREQUEST")
  (make-package "REALSENSE2_CAMERA::DEVICEINFOREQUEST"))
(unless (find-package "REALSENSE2_CAMERA::DEVICEINFORESPONSE")
  (make-package "REALSENSE2_CAMERA::DEVICEINFORESPONSE"))

(in-package "ROS")





(defclass realsense2_camera::DeviceInfoRequest
  :super ros::object
  :slots ())

(defmethod realsense2_camera::DeviceInfoRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass realsense2_camera::DeviceInfoResponse
  :super ros::object
  :slots (_device_name _serial_number _firmware_version _usb_type_descriptor _firmware_update_id _sensors ))

(defmethod realsense2_camera::DeviceInfoResponse
  (:init
   (&key
    ((:device_name __device_name) "")
    ((:serial_number __serial_number) "")
    ((:firmware_version __firmware_version) "")
    ((:usb_type_descriptor __usb_type_descriptor) "")
    ((:firmware_update_id __firmware_update_id) "")
    ((:sensors __sensors) "")
    )
   (send-super :init)
   (setq _device_name (string __device_name))
   (setq _serial_number (string __serial_number))
   (setq _firmware_version (string __firmware_version))
   (setq _usb_type_descriptor (string __usb_type_descriptor))
   (setq _firmware_update_id (string __firmware_update_id))
   (setq _sensors (string __sensors))
   self)
  (:device_name
   (&optional __device_name)
   (if __device_name (setq _device_name __device_name)) _device_name)
  (:serial_number
   (&optional __serial_number)
   (if __serial_number (setq _serial_number __serial_number)) _serial_number)
  (:firmware_version
   (&optional __firmware_version)
   (if __firmware_version (setq _firmware_version __firmware_version)) _firmware_version)
  (:usb_type_descriptor
   (&optional __usb_type_descriptor)
   (if __usb_type_descriptor (setq _usb_type_descriptor __usb_type_descriptor)) _usb_type_descriptor)
  (:firmware_update_id
   (&optional __firmware_update_id)
   (if __firmware_update_id (setq _firmware_update_id __firmware_update_id)) _firmware_update_id)
  (:sensors
   (&optional __sensors)
   (if __sensors (setq _sensors __sensors)) _sensors)
  (:serialization-length
   ()
   (+
    ;; string _device_name
    4 (length _device_name)
    ;; string _serial_number
    4 (length _serial_number)
    ;; string _firmware_version
    4 (length _firmware_version)
    ;; string _usb_type_descriptor
    4 (length _usb_type_descriptor)
    ;; string _firmware_update_id
    4 (length _firmware_update_id)
    ;; string _sensors
    4 (length _sensors)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _device_name
       (write-long (length _device_name) s) (princ _device_name s)
     ;; string _serial_number
       (write-long (length _serial_number) s) (princ _serial_number s)
     ;; string _firmware_version
       (write-long (length _firmware_version) s) (princ _firmware_version s)
     ;; string _usb_type_descriptor
       (write-long (length _usb_type_descriptor) s) (princ _usb_type_descriptor s)
     ;; string _firmware_update_id
       (write-long (length _firmware_update_id) s) (princ _firmware_update_id s)
     ;; string _sensors
       (write-long (length _sensors) s) (princ _sensors s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _device_name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _device_name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _serial_number
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _serial_number (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _firmware_version
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _firmware_version (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _usb_type_descriptor
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _usb_type_descriptor (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _firmware_update_id
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _firmware_update_id (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _sensors
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _sensors (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass realsense2_camera::DeviceInfo
  :super ros::object
  :slots ())

(setf (get realsense2_camera::DeviceInfo :md5sum-) "914e9cfa74a4f66f08c3fe1016943c1b")
(setf (get realsense2_camera::DeviceInfo :datatype-) "realsense2_camera/DeviceInfo")
(setf (get realsense2_camera::DeviceInfo :request) realsense2_camera::DeviceInfoRequest)
(setf (get realsense2_camera::DeviceInfo :response) realsense2_camera::DeviceInfoResponse)

(defmethod realsense2_camera::DeviceInfoRequest
  (:response () (instance realsense2_camera::DeviceInfoResponse :init)))

(setf (get realsense2_camera::DeviceInfoRequest :md5sum-) "914e9cfa74a4f66f08c3fe1016943c1b")
(setf (get realsense2_camera::DeviceInfoRequest :datatype-) "realsense2_camera/DeviceInfoRequest")
(setf (get realsense2_camera::DeviceInfoRequest :definition-)
      "---
string device_name
string serial_number
string firmware_version
string usb_type_descriptor
string firmware_update_id
string sensors

")

(setf (get realsense2_camera::DeviceInfoResponse :md5sum-) "914e9cfa74a4f66f08c3fe1016943c1b")
(setf (get realsense2_camera::DeviceInfoResponse :datatype-) "realsense2_camera/DeviceInfoResponse")
(setf (get realsense2_camera::DeviceInfoResponse :definition-)
      "---
string device_name
string serial_number
string firmware_version
string usb_type_descriptor
string firmware_update_id
string sensors

")



(provide :realsense2_camera/DeviceInfo "914e9cfa74a4f66f08c3fe1016943c1b")


