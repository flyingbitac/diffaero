
(cl:in-package :asdf)

(defsystem "realsense2_camera-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "DeviceInfo" :depends-on ("_package_DeviceInfo"))
    (:file "_package_DeviceInfo" :depends-on ("_package"))
  ))