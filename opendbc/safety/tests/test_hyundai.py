#!/usr/bin/env python3
from opendbc.testing import parameterized_class
import random
import unittest

import numpy as np

from opendbc.car.hyundai.values import CAR, CarControllerParams, HyundaiFlags, HyundaiSafetyFlags
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety
from opendbc.safety.tests.hyundai_common import HyundaiButtonBase, HyundaiLongitudinalBase

from opendbc.sunnypilot.car.hyundai.values import HyundaiSafetyFlagsSP

# LDA button availability
LDA_BUTTON = [
  {"SAFETY_PARAM_SP": HyundaiSafetyFlagsSP.DEFAULT},
  {"SAFETY_PARAM_SP": HyundaiSafetyFlagsSP.HAS_LDA_BUTTON},
]

# All combinations of non-SCC HEV/PHEV/EV cars
_ALL_NON_SCC_HEV_EV_COMBOS = [
  # Hybrid
  {"PCM_STATUS_MSG": ("E_CRUISE_CONTROL", "CRUISE_LAMP_S"),
   "ACC_STATE_MSG": ("E_CRUISE_CONTROL", "CRUISE_LAMP_M"),
   "GAS_MSG": ("E_EMS11", "CR_Vcu_AccPedDep_Pos"),
   "SAFETY_PARAM": HyundaiSafetyFlags.HYBRID_GAS},
  # EV
  {"PCM_STATUS_MSG": ("LABEL11", "CC_ACT"),
   "ACC_STATE_MSG": ("LABEL11", "CC_React"),
   "GAS_MSG": ("E_EMS11", "Accel_Pedal_Pos"),
   "SAFETY_PARAM": HyundaiSafetyFlags.EV_GAS},
]
ALL_NON_SCC_HEV_EV_COMBOS = [{**p, **lda} for lda in LDA_BUTTON for p in _ALL_NON_SCC_HEV_EV_COMBOS]


# 4 bit checkusm used in some hyundai messages
# lives outside the can packer because we never send this msg
def checksum(msg):
  addr, dat, bus = msg

  chksum = 0
  if addr == 0x386:
    for i, b in enumerate(dat):
      for j in range(8):
        # exclude checksum and counter bits
        if (i != 1 or j < 6) and (i != 3 or j < 6) and (i != 5 or j < 6) and (i != 7 or j < 6):
          bit = (b >> j) & 1
        else:
          bit = 0
        chksum += bit
    chksum = (chksum ^ 9) & 0xF
    ret = bytearray(dat)
    ret[5] |= (chksum & 0x3) << 6
    ret[7] |= (chksum & 0xc) << 4
  else:
    for i, b in enumerate(dat):
      if addr in [0x260, 0x421] and i == 7:
        b &= 0x0F if addr == 0x421 else 0xF0
      elif addr == 0x394 and i == 6:
        b &= 0xF0
      elif addr == 0x394 and i == 7:
        continue
      chksum += sum(divmod(b, 16))
    chksum = (16 - chksum) % 16
    ret = bytearray(dat)
    ret[6 if addr == 0x394 else 7] |= chksum << (4 if addr == 0x421 else 0)

  return addr, ret, bus


@parameterized_class(LDA_BUTTON)
class TestHyundaiSafety(HyundaiButtonBase, common.CarSafetyTest, common.DriverTorqueSteeringSafetyTest, common.SteerRequestCutSafetyTest):
  TX_MSGS = [[0x340, 0], [0x4F1, 0], [0x485, 0]]
  STANDSTILL_THRESHOLD = 12  # 0.375 kph
  RELAY_MALFUNCTION_ADDRS = {0: (0x340, 0x485)}  # LKAS11
  FWD_BLACKLISTED_ADDRS = {2: [0x340, 0x485]}

  MAX_RATE_UP = 3
  MAX_RATE_DOWN = 7
  MAX_TORQUE_LOOKUP = [0], [384]
  MAX_RT_DELTA = 112
  DRIVER_TORQUE_ALLOWANCE = 50
  DRIVER_TORQUE_FACTOR = 2

  # Safety around steering req bit
  MIN_VALID_STEERING_FRAMES = 89
  MAX_INVALID_STEERING_FRAMES = 2

  cnt_gas = 0
  cnt_speed = 0
  cnt_brake = 0
  cnt_cruise = 0
  cnt_button = 0

  SAFETY_PARAM_SP: int = 0

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestHyundaiSafety":
      cls.safety = None
      raise unittest.SkipTest

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, 0)
    self.safety.init_tests()

  def _button_msg(self, buttons, main_button=0, bus=0):
    values = {"CF_Clu_CruiseSwState": buttons, "CF_Clu_CruiseSwMain": main_button, "CF_Clu_AliveCnt1": self.cnt_button}
    self.__class__.cnt_button += 1
    return self.packer.make_can_msg_safety("CLU11", bus, values)

  def _user_gas_msg(self, gas):
    values = {"CF_Ems_AclAct": gas, "AliveCounter": self.cnt_gas % 4}
    self.__class__.cnt_gas += 1
    return self.packer.make_can_msg_safety("EMS16", 0, values, fix_checksum=checksum)

  def _user_brake_msg(self, brake):
    values = {"DriverOverride": 2 if brake else random.choice((0, 1, 3)),
              "AliveCounterTCS": self.cnt_brake % 8}
    self.__class__.cnt_brake += 1
    return self.packer.make_can_msg_safety("TCS13", 0, values, fix_checksum=checksum)

  def _speed_msg(self, speed):
    # safety doesn't scale, so undo the scaling
    values = {"WHL_SPD_%s" % s: speed * 0.03125 for s in ["FL", "FR", "RL", "RR"]}
    values["WHL_SPD_AliveCounter_LSB"] = (self.cnt_speed % 16) & 0x3
    values["WHL_SPD_AliveCounter_MSB"] = (self.cnt_speed % 16) >> 2
    self.__class__.cnt_speed += 1
    return self.packer.make_can_msg_safety("WHL_SPD11", 0, values, fix_checksum=checksum)

  def _pcm_status_msg(self, enable):
    values = {"ACCMode": enable, "CR_VSM_Alive": self.cnt_cruise % 16}
    self.__class__.cnt_cruise += 1
    return self.packer.make_can_msg_safety("SCC12", self.SCC_BUS, values, fix_checksum=checksum)

  def _torque_driver_msg(self, torque):
    values = {"CR_Mdps_StrColTq": torque}
    return self.packer.make_can_msg_safety("MDPS12", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"CR_Lkas_StrToqReq": torque, "CF_Lkas_ActToi": steer_req}
    return self.packer.make_can_msg_safety("LKAS11", 0, values)

  def _acc_state_msg(self, enable):
    values = {"MainMode_ACC": enable}
    return self.packer.make_can_msg_safety("SCC11", self.SCC_BUS, values)

  def _lkas_button_msg(self, enabled):
    if self.SAFETY_PARAM_SP & HyundaiSafetyFlagsSP.HAS_LDA_BUTTON:
      values = {"LDA_BTN": enabled}
      return self.packer.make_can_msg_safety("BCM_PO_11", 0, values)
    else:
      raise NotImplementedError

  def _main_cruise_button_msg(self, enabled):
    return self._button_msg(0, enabled)

  def test_pcm_main_cruise_state_availability(self):
    """Test that ACC main state is correctly set when receiving SCC11 (0x420), toggling HYUNDAI_LONG flag.

    Only applicable to SCC-based cars. Non-SCC cars use different messages for ACC state
    and their rx_checks don't include SCC11 after mode reconfiguration.
    """
    if any('NonSCC' in cls.__name__ for cls in type(self).__mro__):
      raise unittest.SkipTest("Non-SCC cars use different ACC state messages, not SCC11")

    prior_safety_mode = self.safety.get_current_safety_mode()
    prior_safety_param = self.safety.get_current_safety_param()
    safety_param_sp = self.SAFETY_PARAM_SP

    for hyundai_longitudinal in (True, False):
      with self.subTest("hyundai_longitudinal", hyundai_longitudinal=hyundai_longitudinal):
        self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
        self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, 0 if hyundai_longitudinal else HyundaiSafetyFlags.LONG)
        for should_turn_acc_main_on in (True, False):
          with self.subTest("acc_main_on", should_turn_acc_main_on=should_turn_acc_main_on):
            self.safety.set_acc_main_on(False)
            self._rx(self._acc_state_msg(should_turn_acc_main_on))
            expected_acc_main = should_turn_acc_main_on and hyundai_longitudinal
            self.assertEqual(expected_acc_main, self.safety.get_acc_main_on())
    self.safety.set_current_safety_param_sp(safety_param_sp)
    self.safety.set_safety_hooks(prior_safety_mode, prior_safety_param)
    self.safety.init_tests()

  def test_enable_control_allowed_with_mads_button(self):
    """Toggle MADS with MADS button, testing HAS_LDA_BUTTON param gating."""
    default_safety_mode = self.safety.get_current_safety_mode()
    default_safety_param = self.safety.get_current_safety_param()
    default_safety_param_sp = self.SAFETY_PARAM_SP

    try:
      self._lkas_button_msg(False)
    except NotImplementedError as err:
      raise unittest.SkipTest("Skipping test because LDA button is not supported") from err

    # CameraSCC rx_checks always include BCM_PO_11 regardless of HAS_LDA_BUTTON param,
    # so we can only test the has_lda_button=True case for CameraSCC.
    camera_scc = bool(default_safety_param & HyundaiSafetyFlags.CAMERA_SCC)
    lda_button_variants = [True] if camera_scc else [True, False]

    try:
      for enable_mads in (True, False):
        with self.subTest("enable_mads", mads_enabled=enable_mads):
          for has_lda_button_param in lda_button_variants:
            with self.subTest("has_lda_button", has_lda_button_param=has_lda_button_param):
              has_lda_button = HyundaiSafetyFlagsSP.HAS_LDA_BUTTON if has_lda_button_param else 0
              sp = (default_safety_param_sp & ~HyundaiSafetyFlagsSP.HAS_LDA_BUTTON) | has_lda_button
              self.safety.set_current_safety_param_sp(sp)
              self.safety.set_safety_hooks(default_safety_mode, default_safety_param)
              self.safety.init_tests()

              self.safety.set_controls_allowed(False)
              self.safety.set_acc_main_on(False)
              self.safety.set_controls_allowed_lateral(False)
              self.safety.set_mads_params(enable_mads, False, False)
              self.assertEqual(enable_mads, self.safety.get_enable_mads())

              self._rx(self._lkas_button_msg(True))
              self._rx(self._lkas_button_msg(False))
              self.assertEqual(enable_mads and has_lda_button_param, self.safety.get_controls_allowed_lateral())
    finally:
      self.safety.set_current_safety_param_sp(default_safety_param_sp)


@parameterized_class(LDA_BUTTON)
class TestHyundaiSafetyAltLimits(TestHyundaiSafety):
  MAX_RATE_UP = 2
  MAX_RATE_DOWN = 3
  MAX_TORQUE_LOOKUP = [0], [270]

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestHyundaiSafetyAltLimits":
      cls.safety = None
      raise unittest.SkipTest

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, HyundaiSafetyFlags.ALT_LIMITS)
    self.safety.init_tests()


@parameterized_class(LDA_BUTTON)
class TestHyundaiSafetyAltLimits2(TestHyundaiSafety):
  MAX_RATE_UP = 2
  MAX_RATE_DOWN = 3
  MAX_TORQUE_LOOKUP = [0], [170]

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestHyundaiSafetyAltLimits2":
      cls.safety = None
      raise unittest.SkipTest

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, HyundaiSafetyFlags.ALT_LIMITS_2)
    self.safety.init_tests()


@parameterized_class(LDA_BUTTON)
class TestHyundaiSafetyDynamicLimits(TestHyundaiSafety):
  """2024-25 Elantra (CN7): speed-scheduled steering torque.

  Measured on comma's own fleet data for this platform (commaSteeringControl,
  2.65M hands-off engaged samples): at STEER_MAX = 384 the car reaches a median 1.97 m/s^2
  of lateral acceleration at full commanded torque below 7 m/s, while the planner asks for
  2.34, so openpilot sits pinned at the ceiling and under-steers through tight turns. By
  18 m/s, 384 already reaches 2.90. The extra authority exists to get closer to the lateral
  acceleration the planner already asks for at low speed, not to exceed it anywhere.

  The lookup is taken from CarControllerParams rather than written out again here, so the
  opendbc gain and this panda ceiling cannot silently diverge -- the inherited
  test_steer_safety_check and test_against_torque_driver then exercise panda against the
  opendbc numbers across the whole speed range.
  """
  MAX_TORQUE_LOOKUP = CarControllerParams.STEER_MAX_LOOKUP_DYNAMIC
  DYNAMIC_MAX_TORQUE = True

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, HyundaiSafetyFlags.DYNAMIC_LIMITS)
    self.safety.init_tests()

  def _reset_speed_measurement(self, speed):
    # the shared dynamic-torque tests speak m/s; Hyundai's _speed_msg takes raw WHL_SPD counts
    super()._reset_speed_measurement(speed * 3.6 / 0.03125)

  def test_vehicle_speed_tracks_wheel_speed(self):
    # the dynamic limit is only as trustworthy as the speed feeding it. Hyundai CAN did not
    # populate vehicle_speed at all before this flag existed
    for speed in (0.0, 5.0, 10.0, 20.0, 30.0):
      self._reset_speed_measurement(speed)
      self.assertAlmostEqual(self.safety.get_vehicle_speed_min(), speed, places=1,
                             msg=f"vehicle_speed did not track wheel speed at {speed} m/s")

  def test_more_torque_at_low_speed_than_high(self):
    # the point of the change: more authority at intersection speed, unchanged on the highway
    low, high = self._get_max_torque(5.0), self._get_max_torque(30.0)
    self.assertGreater(low, high)
    self.assertEqual(low, max(self.MAX_TORQUE_LOOKUP[1]))
    # panda's ceiling on the highway is the static limit plus the shared +1 fudge in
    # steer_torque_cmd_checks. What openpilot commands up there is still exactly 384.
    self.assertEqual(high, min(self.MAX_TORQUE_LOOKUP[1]) + 1)

  def test_torque_ceiling_follows_speed(self):
    # walk the whole schedule and prove the ceiling moves with speed, in both directions
    self.safety.set_controls_allowed(True)
    self._reset_torque_driver_measurement(0)
    for speed in (0.0, 2.0, 5.0, 8.0, 12.0, 16.0, 25.0, 35.0):
      self._reset_speed_measurement(speed)
      max_torque = self._get_max_torque(speed)
      for sign in (-1, 1):
        self._set_prev_torque(max_torque * sign)
        self.assertTrue(self._tx(self._torque_cmd_msg(max_torque * sign)),
                        msg=f"{max_torque * sign} was rejected at {speed} m/s")
        self._set_prev_torque((max_torque + 1) * sign)
        self.assertFalse(self._tx(self._torque_cmd_msg((max_torque + 1) * sign)),
                         msg=f"{(max_torque + 1) * sign} was allowed at {speed} m/s")

  def test_no_speed_measured_yet(self):
    # vehicle_speed resets to 0 on safety mode init, which is the permissive end of the
    # schedule. Nothing can be sent before cruise engages and the RX checks pass, but bound
    # it rather than assume it: the limit is never more than the low-speed value.
    self.safety.init_tests()
    self.assertEqual(self.safety.get_vehicle_speed_min(), 0)

    self.safety.set_controls_allowed(False)
    self.assertFalse(self._tx(self._torque_cmd_msg(1)))

    self.safety.set_controls_allowed(True)
    self._reset_torque_driver_measurement(0)
    max_torque = max(self.MAX_TORQUE_LOOKUP[1])
    self._set_prev_torque(max_torque)
    self.assertTrue(self._tx(self._torque_cmd_msg(max_torque)))
    self._set_prev_torque(max_torque + 1)
    self.assertFalse(self._tx(self._torque_cmd_msg(max_torque + 1)))

  def test_cold_start_cannot_transmit(self):
    # test_no_speed_measured_yet bounds the ceiling at 409 with no speed measured. Bounding it
    # is not the same as showing the window is unreachable, so attack that directly: with the
    # speed still unknown, nothing may be transmitted at all -- not merely nothing above 409.
    self.safety.init_tests()
    self.assertEqual(self.safety.get_vehicle_speed_min(), 0)

    # 0x386 is one of the mode's RX checks, so until it is arriving at frequency the safety
    # tick invalidates the config, and that clears controls in both directions
    self.safety.set_timer(int(2e6))
    self.safety.set_controls_allowed(True)
    self.safety.set_controls_allowed_lateral(True)
    self.safety.safety_tick_current_safety_config()
    self.assertFalse(self.safety.safety_config_valid())
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.get_controls_allowed_lateral())

    for torque in (1, 384, 385, 409, 410):
      self._set_prev_torque(torque)
      self.assertFalse(self._tx(self._torque_cmd_msg(torque)),
                       msg=f"{torque} was transmitted with no speed measured yet")

  def test_stale_speed_cannot_hold_the_low_speed_ceiling(self):
    # the other half of the same worry: measure a low speed, then stop reporting it. panda
    # would keep the permissive 409 while the car accelerates away. The tick has to cut
    # controls rather than let a stale sample hold the ceiling open.
    self._reset_speed_measurement(0.0)
    self.assertEqual(self._get_max_torque(0.0), max(self.MAX_TORQUE_LOOKUP[1]))

    self.safety.set_timer(int(2e6))
    self.safety.set_controls_allowed(True)
    self.safety.set_controls_allowed_lateral(True)
    self.safety.safety_tick_current_safety_config()
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.get_controls_allowed_lateral())
    self._set_prev_torque(max(self.MAX_TORQUE_LOOKUP[1]))
    self.assertFalse(self._tx(self._torque_cmd_msg(max(self.MAX_TORQUE_LOOKUP[1]))),
                     msg="a stale speed sample kept the raised ceiling open")

  def test_only_the_measured_platform_moves(self):
    # this raises a steering torque ceiling; it must not leak onto any other HKG platform
    dynamic = sorted(c for c in CAR if c.config.flags & HyundaiFlags.DYNAMIC_LIMITS)
    self.assertEqual(dynamic, [CAR.HYUNDAI_ELANTRA_2024])

    # hyundaiLegacy shares hyundai_rx_hook and hyundai_tx_hook -- only .init differs -- so the
    # ungated UPDATE_VEHICLE_SPEED added for this schedule runs on legacy platforms too. It is
    # inert there only because none of them carries the flag. Assert that instead of trusting it.
    legacy = [c for c in CAR if c.config.flags & HyundaiFlags.LEGACY]
    self.assertTrue(legacy, "no legacy platforms found; this assertion has stopped testing anything")
    for platform in legacy:
      self.assertFalse(platform.config.flags & HyundaiFlags.DYNAMIC_LIMITS, platform)

    # and above the top breakpoint the gain openpilot applies is bit-for-bit today's value
    xs, ys = self.MAX_TORQUE_LOOKUP
    for speed in (16.0, 20.0, 30.0, 40.0):
      self.assertEqual(round(float(np.interp(speed, xs, ys))), 384)


@parameterized_class(LDA_BUTTON)
class TestHyundaiSafetyCameraSCC(TestHyundaiSafety):
  BUTTONS_TX_BUS = 2  # tx on 2, rx on 0
  SCC_BUS = 2  # rx on 2

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestHyundaiSafetyCameraSCC":
      cls.safety = None
      raise unittest.SkipTest

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, HyundaiSafetyFlags.CAMERA_SCC)
    self.safety.init_tests()

  def test_pcm_main_cruise_state_availability(self):
    """
    Test that ACC main state is correctly set when receiving 0x420 message.
    For camera SCC, ACC main should always be on when receiving 0x420 message
    """

    for should_turn_acc_main_on in (True, False):
      with self.subTest("acc_main_on", should_turn_acc_main_on=should_turn_acc_main_on):
        self._rx(self._acc_state_msg(should_turn_acc_main_on))
        self.assertEqual(should_turn_acc_main_on, self.safety.get_acc_main_on())


@parameterized_class(LDA_BUTTON)
class TestHyundaiSafetyFCEV(TestHyundaiSafety):
  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestHyundaiSafetyFCEV":
      cls.safety = None
      raise unittest.SkipTest

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, HyundaiSafetyFlags.FCEV_GAS)
    self.safety.init_tests()

  def _user_gas_msg(self, gas):
    values = {"ACCELERATOR_PEDAL": gas}
    return self.packer.make_can_msg_safety("FCEV_ACCELERATOR", 0, values)


@parameterized_class(LDA_BUTTON)
class TestHyundaiLegacySafety(TestHyundaiSafety):
  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundaiLegacy, 0)
    self.safety.init_tests()


@parameterized_class(LDA_BUTTON)
class TestHyundaiLegacySafetyEV(TestHyundaiSafety):
  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundaiLegacy, HyundaiSafetyFlags.EV_GAS)
    self.safety.init_tests()

  def _user_gas_msg(self, gas):
    values = {"Accel_Pedal_Pos": gas}
    return self.packer.make_can_msg_safety("E_EMS11", 0, values, fix_checksum=checksum)


@parameterized_class(LDA_BUTTON)
class TestHyundaiLegacySafetyHEV(TestHyundaiSafety):
  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundaiLegacy, HyundaiSafetyFlags.HYBRID_GAS)
    self.safety.init_tests()

  def _user_gas_msg(self, gas):
    values = {"CR_Vcu_AccPedDep_Pos": gas}
    return self.packer.make_can_msg_safety("E_EMS11", 0, values, fix_checksum=checksum)


@parameterized_class(LDA_BUTTON)
class TestHyundaiLongitudinalSafety(HyundaiLongitudinalBase, TestHyundaiSafety):
  TX_MSGS = [[0x340, 0], [0x4F1, 0], [0x485, 0], [0x420, 0], [0x421, 0], [0x50A, 0], [0x389, 0], [0x4A2, 0], [0x38D, 0], [0x483, 0], [0x7D0, 0]]

  FWD_BLACKLISTED_ADDRS = {2: [0x340, 0x485, 0x421, 0x420, 0x50A, 0x389]}

  RELAY_MALFUNCTION_ADDRS = {0: (0x340, 0x485, 0x421, 0x420, 0x50A, 0x389)}  # LKAS11, LFAHDA_MFC, SCC12, SCC11, SCC13, SCC14

  DISABLED_ECU_UDS_MSG = (0x7D0, 0)
  DISABLED_ECU_ACTUATION_MSG = (0x421, 0)

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestHyundaiLongitudinalSafety":
      cls.safety = None
      raise unittest.SkipTest

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, HyundaiSafetyFlags.LONG)
    self.safety.init_tests()

  def _accel_msg(self, accel, aeb_req=False, aeb_decel=0):
    values = {
      "aReqRaw": accel,
      "aReqValue": accel,
      "AEB_CmdAct": int(aeb_req),
      "CR_VSM_DecCmd": aeb_decel,
    }
    return self.packer.make_can_msg_safety("SCC12", self.SCC_BUS, values)

  def _fca11_msg(self, idx=0, vsm_aeb_req=False, fca_aeb_req=False, aeb_decel=0):
    values = {
      "CR_FCA_Alive": idx % 0xF,
      "FCA_Status": 2,
      "CR_VSM_DecCmd": aeb_decel,
      "CF_VSM_DecCmdAct": int(vsm_aeb_req),
      "FCA_CmdAct": int(fca_aeb_req),
    }
    return self.packer.make_can_msg_safety("FCA11", 0, values)

  def _tx_acc_state_msg(self, enable):
    values = {"MainMode_ACC": enable}
    return self.packer.make_can_msg_safety("SCC11", 0, values)

  def test_no_aeb_fca11(self):
    self.assertTrue(self._tx(self._fca11_msg()))
    self.assertFalse(self._tx(self._fca11_msg(vsm_aeb_req=True)))
    self.assertFalse(self._tx(self._fca11_msg(fca_aeb_req=True)))
    self.assertFalse(self._tx(self._fca11_msg(aeb_decel=1.0)))

  def test_no_aeb_scc12(self):
    self.assertTrue(self._tx(self._accel_msg(0)))
    self.assertFalse(self._tx(self._accel_msg(0, aeb_req=True)))
    self.assertFalse(self._tx(self._accel_msg(0, aeb_decel=1.0)))


@parameterized_class(LDA_BUTTON)
class TestHyundaiLongitudinalSafetyCameraSCC(HyundaiLongitudinalBase, TestHyundaiSafety):
  TX_MSGS = [[0x340, 0], [0x4F1, 2], [0x485, 0], [0x420, 0], [0x421, 0], [0x50A, 0], [0x389, 0], [0x4A2, 0]]

  FWD_BLACKLISTED_ADDRS = {2: [0x340, 0x485, 0x420, 0x421, 0x50A, 0x389]}
  RELAY_MALFUNCTION_ADDRS = {0: (0x340, 0x485, 0x421, 0x420, 0x50A, 0x389)}  # LKAS11, LFAHDA_MFC, SCC12, SCC11, SCC13, SCC14

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(HyundaiSafetyFlagsSP.HAS_LDA_BUTTON)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, HyundaiSafetyFlags.LONG | HyundaiSafetyFlags.CAMERA_SCC)
    self.safety.init_tests()

  def _accel_msg(self, accel, aeb_req=False, aeb_decel=0):
    values = {
      "aReqRaw": accel,
      "aReqValue": accel,
      "AEB_CmdAct": int(aeb_req),
      "CR_VSM_DecCmd": aeb_decel,
    }
    return self.packer.make_can_msg_safety("SCC12", self.SCC_BUS, values)

  def _tx_acc_state_msg(self, enable):
    values = {"MainMode_ACC": enable}
    return self.packer.make_can_msg_safety("SCC11", self.SCC_BUS, values)

  def test_no_aeb_scc12(self):
    self.assertTrue(self._tx(self._accel_msg(0)))
    self.assertFalse(self._tx(self._accel_msg(0, aeb_req=True)))
    self.assertFalse(self._tx(self._accel_msg(0, aeb_decel=1.0)))

  def test_tester_present_allowed(self):
    pass

  def test_disabled_ecu_alive(self):
    pass


@parameterized_class(LDA_BUTTON)
class TestHyundaiSafetyFCEVLong(TestHyundaiLongitudinalSafety, TestHyundaiSafetyFCEV):
  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestHyundaiSafetyFCEVLong":
      cls.safety = None
      raise unittest.SkipTest

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, HyundaiSafetyFlags.FCEV_GAS | HyundaiSafetyFlags.LONG)
    self.safety.init_tests()


@parameterized_class(LDA_BUTTON)
class TestHyundaiLongitudinalESCCSafety(HyundaiLongitudinalBase, TestHyundaiSafety):
  TX_MSGS = [[0x340, 0], [0x4F1, 0], [0x485, 0], [0x420, 0], [0x421, 0], [0x50A, 0], [0x389, 0]]

  FWD_BLACKLISTED_ADDRS = {2: [0x340, 0x485, 0x420, 0x421, 0x50A, 0x389]}
  RELAY_MALFUNCTION_ADDRS = {0: (0x340, 0x485, 0x420, 0x421, 0x50A, 0x389)}  # LKAS11, LFAHDA_MFC, SCC12, SCC11, SCC13, SCC14

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestHyundaiLongitudinalESCCSafety":
      cls.safety = None
      raise unittest.SkipTest

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(HyundaiSafetyFlagsSP.ESCC | self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, HyundaiSafetyFlags.LONG)
    self.safety.init_tests()

  def _accel_msg(self, accel, aeb_req=False, aeb_decel=0):
    values = {
      "aReqRaw": accel,
      "aReqValue": accel,
    }
    return self.packer.make_can_msg_safety("SCC12", self.SCC_BUS, values)

  def _tx_acc_state_msg(self, enable):
    values = {"MainMode_ACC": enable}
    return self.packer.make_can_msg_safety("SCC11", 0, values)

  def test_tester_present_allowed(self):
    pass

  def test_disabled_ecu_alive(self):
    pass


@parameterized_class(LDA_BUTTON)
class TestHyundaiNonSCCSafety(TestHyundaiSafety):

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestHyundaiNonSCCSafety":
      cls.safety = None
      raise unittest.SkipTest

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(HyundaiSafetyFlagsSP.NON_SCC | self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, 0)
    self.safety.init_tests()

  def _pcm_status_msg(self, enable):
    values = {"CRUISE_LAMP_S": enable, "AliveCounter": self.cnt_gas % 4}
    self.__class__.cnt_gas += 1
    return self.packer.make_can_msg_safety("EMS16", 0, values, fix_checksum=checksum)

  def _acc_state_msg(self, enable):
    values = {"CRUISE_LAMP_M": enable, "AliveCounter": self.cnt_gas % 4}
    self.__class__.cnt_gas += 1
    return self.packer.make_can_msg_safety("EMS16", 0, values, fix_checksum=checksum)

  def _user_gas_msg(self, gas: float, controls_allowed: bool = True):
    values = {"CF_Ems_AclAct": gas, "CRUISE_LAMP_M": 1, "CRUISE_LAMP_S": controls_allowed, "AliveCounter": self.cnt_gas % 4}
    self.__class__.cnt_gas += 1
    return self.packer.make_can_msg_safety("EMS16", 0, values, fix_checksum=checksum)

  def test_allow_engage_with_gas_pressed(self):
    self._rx(self._user_gas_msg(1, self.safety.get_controls_allowed()))
    self.safety.set_controls_allowed(True)
    self._rx(self._user_gas_msg(1, self.safety.get_controls_allowed()))
    self.assertTrue(self.safety.get_controls_allowed())
    self._rx(self._user_gas_msg(1, self.safety.get_controls_allowed()))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_no_disengage_on_gas(self):
    self._rx(self._user_gas_msg(0, self.safety.get_controls_allowed()))
    self.safety.set_controls_allowed(True)
    self._rx(self._user_gas_msg(self.GAS_PRESSED_THRESHOLD + 1, self.safety.get_controls_allowed()))
    # Test we allow lateral, but not longitudinal
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.get_longitudinal_allowed())
    # Make sure we can re-gain longitudinal actuation
    self._rx(self._user_gas_msg(0, self.safety.get_controls_allowed()))
    self.assertTrue(self.safety.get_longitudinal_allowed())


@parameterized_class(ALL_NON_SCC_HEV_EV_COMBOS)
class TestHyundaiNonSCCSafety_HEV_EV(TestHyundaiSafety):

  PCM_STATUS_MSG = ("", "")
  ACC_STATE_MSG = ("", "")
  GAS_MSG = ("", "")
  SAFETY_PARAM = 0

  @classmethod
  def setUpClass(cls):
    if cls.__name__ == "TestHyundaiNonSCCSafety_HEV_EV":
      cls.safety = None
      raise unittest.SkipTest

  def setUp(self):
    self.packer = CANPackerSafety("hyundai_can_generated")
    self.safety = libsafety_py.libsafety
    self.safety.set_current_safety_param_sp(HyundaiSafetyFlagsSP.NON_SCC | self.SAFETY_PARAM_SP)
    self.safety.set_safety_hooks(CarParams.SafetyModel.hyundai, self.SAFETY_PARAM)
    self.safety.init_tests()

  def _pcm_status_msg(self, enable):
    values = {self.PCM_STATUS_MSG[1]: enable}
    return self.packer.make_can_msg_safety(self.PCM_STATUS_MSG[0], 0, values)

  def _acc_state_msg(self, enable):
    values = {self.ACC_STATE_MSG[1]: enable}
    return self.packer.make_can_msg_safety(self.ACC_STATE_MSG[0], 0, values)

  def _user_gas_msg(self, gas):
    values = {self.GAS_MSG[1]: gas}
    return self.packer.make_can_msg_safety(self.GAS_MSG[0], 0, values, fix_checksum=checksum)


if __name__ == "__main__":
  unittest.main()
