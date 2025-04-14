// Auto-generated. Do not edit!

// (in-package realsense2_camera.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class DeviceInfoRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type DeviceInfoRequest
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type DeviceInfoRequest
    let len;
    let data = new DeviceInfoRequest(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'realsense2_camera/DeviceInfoRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd41d8cd98f00b204e9800998ecf8427e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new DeviceInfoRequest(null);
    return resolved;
    }
};

class DeviceInfoResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.device_name = null;
      this.serial_number = null;
      this.firmware_version = null;
      this.usb_type_descriptor = null;
      this.firmware_update_id = null;
      this.sensors = null;
    }
    else {
      if (initObj.hasOwnProperty('device_name')) {
        this.device_name = initObj.device_name
      }
      else {
        this.device_name = '';
      }
      if (initObj.hasOwnProperty('serial_number')) {
        this.serial_number = initObj.serial_number
      }
      else {
        this.serial_number = '';
      }
      if (initObj.hasOwnProperty('firmware_version')) {
        this.firmware_version = initObj.firmware_version
      }
      else {
        this.firmware_version = '';
      }
      if (initObj.hasOwnProperty('usb_type_descriptor')) {
        this.usb_type_descriptor = initObj.usb_type_descriptor
      }
      else {
        this.usb_type_descriptor = '';
      }
      if (initObj.hasOwnProperty('firmware_update_id')) {
        this.firmware_update_id = initObj.firmware_update_id
      }
      else {
        this.firmware_update_id = '';
      }
      if (initObj.hasOwnProperty('sensors')) {
        this.sensors = initObj.sensors
      }
      else {
        this.sensors = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type DeviceInfoResponse
    // Serialize message field [device_name]
    bufferOffset = _serializer.string(obj.device_name, buffer, bufferOffset);
    // Serialize message field [serial_number]
    bufferOffset = _serializer.string(obj.serial_number, buffer, bufferOffset);
    // Serialize message field [firmware_version]
    bufferOffset = _serializer.string(obj.firmware_version, buffer, bufferOffset);
    // Serialize message field [usb_type_descriptor]
    bufferOffset = _serializer.string(obj.usb_type_descriptor, buffer, bufferOffset);
    // Serialize message field [firmware_update_id]
    bufferOffset = _serializer.string(obj.firmware_update_id, buffer, bufferOffset);
    // Serialize message field [sensors]
    bufferOffset = _serializer.string(obj.sensors, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type DeviceInfoResponse
    let len;
    let data = new DeviceInfoResponse(null);
    // Deserialize message field [device_name]
    data.device_name = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [serial_number]
    data.serial_number = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [firmware_version]
    data.firmware_version = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [usb_type_descriptor]
    data.usb_type_descriptor = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [firmware_update_id]
    data.firmware_update_id = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [sensors]
    data.sensors = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.device_name);
    length += _getByteLength(object.serial_number);
    length += _getByteLength(object.firmware_version);
    length += _getByteLength(object.usb_type_descriptor);
    length += _getByteLength(object.firmware_update_id);
    length += _getByteLength(object.sensors);
    return length + 24;
  }

  static datatype() {
    // Returns string type for a service object
    return 'realsense2_camera/DeviceInfoResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '914e9cfa74a4f66f08c3fe1016943c1b';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string device_name
    string serial_number
    string firmware_version
    string usb_type_descriptor
    string firmware_update_id
    string sensors
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new DeviceInfoResponse(null);
    if (msg.device_name !== undefined) {
      resolved.device_name = msg.device_name;
    }
    else {
      resolved.device_name = ''
    }

    if (msg.serial_number !== undefined) {
      resolved.serial_number = msg.serial_number;
    }
    else {
      resolved.serial_number = ''
    }

    if (msg.firmware_version !== undefined) {
      resolved.firmware_version = msg.firmware_version;
    }
    else {
      resolved.firmware_version = ''
    }

    if (msg.usb_type_descriptor !== undefined) {
      resolved.usb_type_descriptor = msg.usb_type_descriptor;
    }
    else {
      resolved.usb_type_descriptor = ''
    }

    if (msg.firmware_update_id !== undefined) {
      resolved.firmware_update_id = msg.firmware_update_id;
    }
    else {
      resolved.firmware_update_id = ''
    }

    if (msg.sensors !== undefined) {
      resolved.sensors = msg.sensors;
    }
    else {
      resolved.sensors = ''
    }

    return resolved;
    }
};

module.exports = {
  Request: DeviceInfoRequest,
  Response: DeviceInfoResponse,
  md5sum() { return '914e9cfa74a4f66f08c3fe1016943c1b'; },
  datatype() { return 'realsense2_camera/DeviceInfo'; }
};
