import re
import unittest
from collections import defaultdict
from pathlib import Path

from opendbc.can import CANPacker
from opendbc.car import gen_empty_fingerprint, Bus, structs
from opendbc.car.structs import CarParams
from opendbc.car.fw_versions import build_fw_dict
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.hyundaicanfd import CanBus
from opendbc.car.hyundai.radar_interface import RADAR_START_ADDR
from opendbc.car.hyundai.values import CAR, DATE_FW_ECUS, DBC, \
                                         FW_QUERY_CONFIG, CANFD_FUZZY_WHITELIST, \
                                         PLATFORM_CODE_ECUS, HYUNDAI_VERSION_REQUEST_LONG, \
                                         HyundaiFlags, get_platform_codes, HyundaiSafetyFlags, \
                                         NON_SCC_CAR, CarControllerParams
from opendbc.car.hyundai.fingerprints import FW_VERSIONS
from opendbc.testing import fuzzy_test

Ecu = CarParams.Ecu

# Some platforms have date codes in a different format we don't yet parse (or are missing).
# For now, assert list of expected missing date cars
NO_DATES_PLATFORMS = {
  # CAN FD
  CAR.KIA_SPORTAGE_5TH_GEN,
  CAR.HYUNDAI_SANTA_CRUZ_1ST_GEN,
  CAR.HYUNDAI_TUCSON_4TH_GEN,
  # CAN
  CAR.HYUNDAI_ELANTRA,
  CAR.HYUNDAI_ELANTRA_GT_I30,
  CAR.KIA_CEED,
  CAR.KIA_CEED_PHEV,
  CAR.KIA_FORTE,
  CAR.KIA_OPTIMA_G4,
  CAR.KIA_OPTIMA_G4_FL,
  CAR.KIA_SORENTO,
  CAR.HYUNDAI_KONA,
  CAR.HYUNDAI_KONA_EV,
  CAR.HYUNDAI_KONA_EV_2022,
  CAR.HYUNDAI_KONA_HEV,
  CAR.HYUNDAI_SONATA_LF,
  CAR.HYUNDAI_VELOSTER,
  CAR.HYUNDAI_KONA_2022,
}

CANFD_EXPECTED_ECUS = {Ecu.fwdCamera, Ecu.fwdRadar}

# CAN-only feature flags that should not appear on CAN FD platforms
CAN_FEATURE_FLAGS = (HyundaiFlags.CLUSTER_GEARS | HyundaiFlags.TCU_GEARS | HyundaiFlags.CHECKSUM_CRC8 |
                     HyundaiFlags.CHECKSUM_6B | HyundaiFlags.LEGACY | HyundaiFlags.UNSUPPORTED_LONGITUDINAL |
                     HyundaiFlags.CAMERA_SCC | HyundaiFlags.RAISED_LIMITS)

# The only platforms allowed to command more than the 384-count HKG default. Stated as an
# explicit set so that widening the raised ceiling to another car is a deliberate, reviewed
# edit to this line rather than a side effect of adding a flag somewhere else.
RAISED_LIMITS_PLATFORMS = {CAR.HYUNDAI_ELANTRA_2024}

# The ceiling is FLAT: 409 at every speed, and there is no mechanism by which it could be
# anything else -- CarControllerParams carries no lookup and the carcontroller is byte-for-byte
# upstream's. 409 is where the CN7's EPS stops faulting (500 and 450 both threw a fault on the
# car), so this is a measured hardware boundary, not a tuning choice.
RAISED_STEER_MAX = 409


def panda_raised_ceiling():
  # Read panda's ceiling out of the C source rather than copying the number. opendbc commands
  # 409 and panda enforces 512, so the two are deliberately NOT equal -- a hand-copied
  # constant here would agree with itself while the two drifted apart.
  src = Path(__file__).parents[3] / "safety" / "modes" / "hyundai.h"
  # Anchored to a real DECLARATION, at the start of a line. An unanchored search takes the first
  # match anywhere in the file, so a stale value left in a comment above the live line -- exactly
  # what someone writes when lowering it -- is read instead of the value in force, and this gate
  # then passes while panda drops every frame opendbc commands above the real ceiling. It
  # fails open in one direction only, which is why the reverse-direction decoy did not expose it.
  pattern = (r"^\s*const\s+TorqueSteeringLimits\s+HYUNDAI_STEERING_LIMITS_RAISED\s*=" +
             r"\s*HYUNDAI_LIMITS\(\s*(\d+)")
  m = re.search(pattern, src.read_text(), re.MULTILINE)
  assert m, f"could not find HYUNDAI_STEERING_LIMITS_RAISED in {src}"
  return int(m.group(1))


def cars_with(flags):
  return {c for c in CAR if c.config.flags & flags}


class TestHyundaiFingerprint(unittest.TestCase):
  def test_feature_detection(self):
    # LKA steering
    for lka_steering in (True, False):
      fingerprint = gen_empty_fingerprint()
      if lka_steering:
        cam_can = CanBus(None, fingerprint).CAM
        fingerprint[cam_can] = [0x50, 0x110]  # LKA steering messages
      CP = CarInterface.get_params(CAR.KIA_EV6, fingerprint, [], False, False, False)
      assert bool(CP.flags & HyundaiFlags.CANFD_LKA_STEER_MSG) == lka_steering

    # radar available
    for radar in (True, False):
      fingerprint = gen_empty_fingerprint()
      if radar:
        fingerprint[1][RADAR_START_ADDR] = 8
      CP = CarInterface.get_params(CAR.HYUNDAI_SONATA, fingerprint, [], False, False, False)
      assert CP.radarUnavailable != radar

  def test_alternate_limits(self):
    # Alternate lateral control limits, for high torque cars, verify Panda safety mode flag is set
    fingerprint = gen_empty_fingerprint()
    for car_model in CAR:
      CP = CarInterface.get_params(car_model, fingerprint, [], False, False, False)
      assert bool(CP.flags & HyundaiFlags.ALT_LIMITS) == bool(CP.safetyConfigs[-1].safetyParam & HyundaiSafetyFlags.ALT_LIMITS)

  def test_raised_limits(self):
    # The raised ceiling lives in four places that cannot see each other: the platform flag,
    # the safetyParam bit panda reads, CarControllerParams.STEER_MAX, and the list above.
    # This executes the interface.py bridge rather than reading its source, so moving that
    # line into a branch that never runs for this platform is caught too.
    fingerprint = gen_empty_fingerprint()
    carrying = set()
    for car_model in CAR:
      CP = CarInterface.get_params(car_model, fingerprint, [], False, False, False)
      car_flag = bool(CP.flags & HyundaiFlags.RAISED_LIMITS)
      safety_bit = bool(CP.safetyConfigs[-1].safetyParam & HyundaiSafetyFlags.RAISED_LIMITS)
      steer_max = CarControllerParams(CP).STEER_MAX

      # without this openpilot commands 409 and panda rejects every frame above 384
      assert car_flag == safety_bit, f"{car_model}: flag/safetyParam disagree ({car_flag} vs {safety_bit})"
      # and the flag has to be what actually moves the number. The ceiling is flat, so this
      # is the whole of it; test_no_hyundai_carries_a_torque_schedule proves no schedule exists.
      assert (steer_max == RAISED_STEER_MAX) == car_flag, f"{car_model}: STEER_MAX {steer_max} with flag {car_flag}"
      if car_flag:
        carrying.add(car_model)
        # only platforms whose stock ceiling is the 384 default may be raised; a car also
        # carrying CANFD/ALT_LIMITS/ALT_LIMITS_2 would command 409 where panda enforces less
        lower = HyundaiFlags.CANFD | HyundaiFlags.ALT_LIMITS | HyundaiFlags.ALT_LIMITS_2
        assert not (CP.flags & lower), f"{car_model}: raised ceiling on a car panda limits lower"

    assert carrying == RAISED_LIMITS_PLATFORMS, f"raised ceiling on {carrying}, expected {RAISED_LIMITS_PLATFORMS}"

  def test_no_hyundai_carries_a_torque_schedule(self):
    # A speed-scheduled ceiling was built and driven here, and the EPS faulted at both 500 and
    # 450. It is not coming back, so what this asserts is that the MECHANISM is absent: no
    # lookup attribute, and a carcontroller with no per-frame ceiling. Text-level, because an
    # attribute that does not exist cannot be swept for.
    #
    # This is deliberately cheap. The expensive per-platform sweep it replaced could only ever
    # prove that a thing which does not exist did not do anything.
    values = (Path(__file__).parents[1] / "values.py").read_text()
    carctl = (Path(__file__).parents[1] / "carcontroller.py").read_text()
    for dead in ("STEER_MAX_LOOKUP", "steer_max_at"):
      assert dead not in values, f"values.py carries schedule machinery: {dead}"
      assert dead not in carctl, f"carcontroller.py carries schedule machinery: {dead}"
    assert "actuators.torque * self.params.STEER_MAX" in carctl,       "carcontroller no longer multiplies by the static ceiling"

  def test_the_raised_ceiling_sits_under_what_panda_passes(self):
    # The raised platform's single ceiling against panda's, read out of hyundai.h rather than
    # copied so the two cannot drift together. They are deliberately NOT equal: opendbc
    # commands 409, panda enforces 512, and the 103 counts between them are unenforced.
    #
    # The literal 409 is asserted as well as the relationship, and that matters more than it
    # used to. Panda's 512 sits above the band the EPS faults in, so "at or under panda" would
    # happily pass a 450 that stops the car steering. The literal is the real check.
    ceiling = panda_raised_ceiling()
    fingerprint = gen_empty_fingerprint()

    for car_model in CAR:
      CP = CarInterface.get_params(car_model, fingerprint, [], False, False, False)
      if CP.flags & HyundaiFlags.RAISED_LIMITS:
        params = CarControllerParams(CP)
        assert params.STEER_MAX == RAISED_STEER_MAX
        assert params.STEER_MAX <= ceiling,           f"{car_model}: commands {params.STEER_MAX} where panda enforces {ceiling}"

  def test_the_ceiling_reaches_the_wire(self):
    # Everything above reads source. This one RUNS the car: it builds the real CarInterface,
    # drives the real CarController for long enough that the 3-count-per-frame rate limiter is
    # no longer what binds, and decodes CR_Lkas_StrToqReq out of the LKAS11 frame the car would
    # actually transmit.
    #
    # This exists because the source-text guards cannot see wiring. Every check above can stay
    # green while the number that reaches CAN is something else entirely -- that exact bug
    # shipped once on this branch (the interface.py flag bridge that nothing executed). It is
    # the only check that proves 409 is what the car sends.
    #
    # The speed sweep is kept even though the ceiling is flat: a flat ceiling is a CLAIM about
    # every speed, and this is what tests it. If a lookup ever comes back, the wire changes
    # here first.
    fp = gen_empty_fingerprint()
    CP = CarInterface.get_params(CAR.HYUNDAI_ELANTRA_2024, fp, [], alpha_long=False,
                                 is_release=False, docs=False)
    CP_SP = CarInterface.get_params_sp(CP, CAR.HYUNDAI_ELANTRA_2024, fp, [], alpha_long=False,
                                       is_release_sp=False, docs=False)

    def drive(v_ego_raw, frames=400):
      CI = CarInterface(CP, CP_SP)
      CI.CS.out = structs.CarState(vEgoRaw=v_ego_raw, steeringTorque=0.0, steeringAngleDeg=0.0)
      # the stock signals the carcontroller copies through; zeros, because nothing here reads
      # them back. defaultdict rather than a literal list so this does not need editing when
      # create_lkas11's passthrough set changes.
      CI.CS.lkas11 = defaultdict(int)
      CI.CS.clu11 = defaultdict(int)
      CI.CS.is_metric = False
      CI.CS.main_cruise_enabled = False

      CC = structs.CarControl(enabled=True, latActive=True)
      CC.actuators.torque = 1.0        # full demand, so the ceiling is what binds
      CC = CC.as_reader()
      CC_SP = structs.CarControlSP()

      on_wire, actuators = None, None
      for i in range(frames):
        actuators, can_sends = CI.apply(CC, CC_SP, i * 10_000_000)
        for addr, dat, _bus in can_sends:
          if addr == 0x340:            # LKAS11
            word = int.from_bytes(bytes(dat)[0:4], "little")
            on_wire = ((word >> 16) & 0x7FF) - 1024      # CR_Lkas_StrToqReq, 16|11@1+ (1,-1024)
      return on_wire, actuators

    # Includes the breakpoints the retired schedule used (8.94 and 13.41 m/s) precisely because
    # they are where a reintroduced lookup would bend. Under a flat ceiling they are ordinary
    # speeds and must be indistinguishable from the rest.
    for v in (0.0, 5.0, 8.94, 11.175, 13.41, 25.0, 40.0):
      on_wire, actuators = drive(v)

      assert on_wire == RAISED_STEER_MAX,         f"at {v} m/s the wire carries {on_wire}, expected a flat {RAISED_STEER_MAX}"
      assert actuators.torqueOutputCan == RAISED_STEER_MAX,         f"at {v} m/s torqueOutputCan is {actuators.torqueOutputCan}, expected {RAISED_STEER_MAX}"

      # And the feedback normalisation must divide by the ceiling the command was built from.
      # Saturated, that is 1.0 exactly. Dividing by anything else feeds the torque learner a
      # demand that never existed.
      assert abs(actuators.torque - 1.0) < 1e-9,         f"at {v} m/s normalised feedback is {actuators.torque}, not 1.0"

  def test_lfahda_mfc_length(self):
    # Three things have to agree per platform and none of them can see the others: the dbc
    # (what openpilot packs into 0x485), the safetyParam bit (what panda will let out), and
    # the car flag that is supposed to drive both. Disagreement is either a car that cannot
    # show HDA state or a panda allow-list wider than the message it guards, and neither
    # shows up as a test failure anywhere else.
    fingerprint = gen_empty_fingerprint()
    for car_model in CAR:
      CP = CarInterface.get_params(car_model, fingerprint, [], False, False, False)
      if CP.flags & HyundaiFlags.CANFD:
        continue  # LFAHDA_MFC is a CAN-only message

      car_flag = bool(CP.flags & HyundaiFlags.LFAHDA_MFC_8)
      safety_bit = bool(CP.safetyConfigs[-1].safetyParam & HyundaiSafetyFlags.LFAHDA_MFC_8)
      assert car_flag == safety_bit, f"{car_model}: flag/safetyParam disagree"

      packed = CANPacker(DBC[car_model][Bus.pt]).make_can_msg("LFAHDA_MFC", 0, {})[1]
      expected = 8 if car_flag else 4
      assert len(packed) == expected,              f"{car_model}: dbc packs {len(packed)} bytes, flag says {expected}"

  def test_can_features(self):
    for car_model in CAR:
      flags = car_model.config.flags
      # Test no EV/HEV with cluster/TCU gears (should all use ELECT_GEAR)
      if flags & (HyundaiFlags.HYBRID | HyundaiFlags.EV):
        assert not (flags & (HyundaiFlags.CLUSTER_GEARS | HyundaiFlags.TCU_GEARS))

      # Test CAN FD car not in CAN feature lists
      if flags & HyundaiFlags.CANFD:
        assert not (flags & CAN_FEATURE_FLAGS), "CAN FD car unexpectedly has a CAN feature flag"

  def test_hybrid_ev_flags(self):
    for car_model in CAR:
      flags = car_model.config.flags
      assert not (flags & HyundaiFlags.HYBRID and flags & HyundaiFlags.EV), "Shared cars between hybrid and EV"
      assert not (flags & HyundaiFlags.CANFD and flags & HyundaiFlags.HYBRID), \
        "Hard coding CAN FD cars as hybrid is no longer supported"

  def test_canfd_ecu_whitelist(self):
    # Asserts only expected Ecus can exist in database for CAN-FD cars
    for car_model in CAR:
      if not (car_model.config.flags & HyundaiFlags.CANFD):
        continue
      ecus = {fw[0] for fw in FW_VERSIONS[car_model].keys()}
      ecus_not_in_whitelist = ecus - CANFD_EXPECTED_ECUS
      ecu_strings = ", ".join([f"Ecu.{ecu}" for ecu in ecus_not_in_whitelist])
      assert len(ecus_not_in_whitelist) == 0, \
                       f"{car_model}: Car model has unexpected ECUs: {ecu_strings}"

  def test_blacklisted_parts(self):
    # Asserts no ECUs known to be shared across platforms exist in the database.
    # Tucson having Santa Cruz camera and EPS for example
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model.value):
        if car_model == CAR.HYUNDAI_SANTA_CRUZ_1ST_GEN:
          raise unittest.SkipTest("Skip checking Santa Cruz for its parts")

        for code, _ in get_platform_codes(ecus[(Ecu.fwdCamera, 0x7c4, None)]):
          if b"-" not in code:
            continue
          part = code.split(b"-")[1]
          assert not part.startswith(b'CW'), "Car has bad part number"

  def test_correct_ecu_response_database(self):
    """
    Assert standard responses for certain ECUs, since they can
    respond to multiple queries with different data
    """
    expected_fw_prefix = HYUNDAI_VERSION_REQUEST_LONG[1:]
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model.value):
        for ecu, fws in ecus.items():
          assert all(fw.startswith(expected_fw_prefix) for fw in fws), \
                          f"FW from unexpected request in database: {(ecu, fws)}"

  @fuzzy_test(max_examples=100)
  def test_platform_codes_fuzzy_fw(self, fuzzy):
    """Ensure function doesn't raise an exception"""
    get_platform_codes(fuzzy.list(fuzzy.binary))

  def test_expected_platform_codes(self):
    # Ensures we don't accidentally add multiple platform codes for a car unless it is intentional
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model.value):
        for ecu, fws in ecus.items():
          if ecu[0] not in PLATFORM_CODE_ECUS:
            continue

          # Third and fourth character are usually EV/hybrid identifiers
          codes = {code.split(b"-")[0][:2] for code, _ in get_platform_codes(fws)}
          if car_model == CAR.HYUNDAI_PALISADE:
            assert codes == {b"LX", b"ON"}, f"Car has unexpected platform codes: {car_model} {codes}"
          elif car_model == CAR.HYUNDAI_KONA_EV and ecu[0] == Ecu.fwdCamera:
            assert codes == {b"OE", b"OS"}, f"Car has unexpected platform codes: {car_model} {codes}"
          else:
            assert len(codes) == 1, f"Car has multiple platform codes: {car_model} {codes}"

  # Tests for platform codes, part numbers, and FW dates which Hyundai will use to fuzzy
  # fingerprint in the absence of full FW matches:
  def test_platform_code_ecus_available(self):
    # TODO: add queries for these non-CAN FD cars to get EPS
    no_eps_platforms = cars_with(HyundaiFlags.CANFD) | {CAR.KIA_SORENTO, CAR.KIA_OPTIMA_G4, CAR.KIA_OPTIMA_G4_FL, CAR.KIA_OPTIMA_H, CAR.KIA_K7_2017,
                                    CAR.KIA_OPTIMA_H_G4_FL, CAR.HYUNDAI_SONATA_LF, CAR.HYUNDAI_TUCSON, CAR.GENESIS_G90, CAR.GENESIS_G80, CAR.HYUNDAI_ELANTRA}

    # Asserts ECU keys essential for fuzzy fingerprinting are available on all platforms
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model.value):
        for platform_code_ecu in PLATFORM_CODE_ECUS:
          if platform_code_ecu in (Ecu.fwdRadar, Ecu.eps) and car_model == CAR.HYUNDAI_GENESIS:
            continue
          if platform_code_ecu == Ecu.eps and car_model in no_eps_platforms:
            continue
          if car_model in NON_SCC_CAR:
            continue
          assert platform_code_ecu in [e[0] for e in ecus]

  def test_fw_format(self):
    # Asserts:
    # - every supported ECU FW version returns one platform code
    # - every supported ECU FW version has a part number
    # - expected parsing of ECU FW dates

    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model.value):
        for ecu, fws in ecus.items():
          if ecu[0] not in PLATFORM_CODE_ECUS:
            continue

          if car_model in NON_SCC_CAR:
            continue

          codes = set()
          for fw in fws:
            result = get_platform_codes([fw])
            assert 1 == len(result), f"Unable to parse FW: {fw}"
            codes |= result

          if ecu[0] not in DATE_FW_ECUS or car_model in NO_DATES_PLATFORMS:
            assert all(date is None for _, date in codes)
          else:
            assert all(date is not None for _, date in codes)

          if car_model == CAR.HYUNDAI_GENESIS:
            raise unittest.SkipTest("No part numbers for car model")

          # Hyundai places the ECU part number in their FW versions, assert all parsable
          # Some examples of valid formats: b"56310-L0010", b"56310L0010", b"56310/M6300"
          assert all(b"-" in code for code, _ in codes), \
                          f"FW does not have part number: {fw}"

  def test_platform_codes_spot_check(self):
    # Asserts basic platform code parsing behavior for a few cases
    results = get_platform_codes([b"\xf1\x00DH LKAS 1.1 -150210"])
    assert results == {(b"DH", b"150210")}

    # Some cameras and all radars do not have dates
    results = get_platform_codes([b"\xf1\x00AEhe SCC H-CUP      1.01 1.01 96400-G2000         "])
    assert results == {(b"AEhe-G2000", None)}

    results = get_platform_codes([b"\xf1\x00CV1_ RDR -----      1.00 1.01 99110-CV000         "])
    assert results == {(b"CV1-CV000", None)}

    results = get_platform_codes([
      b"\xf1\x00DH LKAS 1.1 -150210",
      b"\xf1\x00AEhe SCC H-CUP      1.01 1.01 96400-G2000         ",
      b"\xf1\x00CV1_ RDR -----      1.00 1.01 99110-CV000         ",
    ])
    assert results == {(b"DH", b"150210"), (b"AEhe-G2000", None), (b"CV1-CV000", None)}

    results = get_platform_codes([
      b"\xf1\x00LX2 MFC  AT USA LHD 1.00 1.07 99211-S8100 220222",
      b"\xf1\x00LX2 MFC  AT USA LHD 1.00 1.08 99211-S8100 211103",
      b"\xf1\x00ON  MFC  AT USA LHD 1.00 1.01 99211-S9100 190405",
      b"\xf1\x00ON  MFC  AT USA LHD 1.00 1.03 99211-S9100 190720",
    ])
    assert results == {(b"LX2-S8100", b"220222"), (b"LX2-S8100", b"211103"),
                               (b"ON-S9100", b"190405"), (b"ON-S9100", b"190720")}

  def test_fuzzy_excluded_platforms(self):
    # Asserts a list of platforms that will not fuzzy fingerprint with platform codes due to them being shared.
    # This list can be shrunk as we combine platforms and detect features
    excluded_platforms = {
      CAR.GENESIS_G70,            # shared platform code, part number, and date
      CAR.GENESIS_G70_2020,
    }
    excluded_platforms |= cars_with(HyundaiFlags.CANFD) - cars_with(HyundaiFlags.EV) - CANFD_FUZZY_WHITELIST  # shared platform codes
    excluded_platforms |= NO_DATES_PLATFORMS  # date codes are required to match

    platforms_with_shared_codes = set()
    for platform, fw_by_addr in FW_VERSIONS.items():
      car_fw = []
      for ecu, fw_versions in fw_by_addr.items():
        ecu_name, addr, sub_addr = ecu
        for fw in fw_versions:
          car_fw.append(CarParams.CarFw(ecu=ecu_name, fwVersion=fw, address=addr,
                                        subAddress=0 if sub_addr is None else sub_addr))

      if platform in NON_SCC_CAR:
        continue

      CP = CarParams(carFw=car_fw)
      matches = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(build_fw_dict(CP.carFw), CP.carVin, FW_VERSIONS)
      if len(matches) == 1:
        assert list(matches)[0] == platform
      else:
        platforms_with_shared_codes.add(platform)

    assert platforms_with_shared_codes == excluded_platforms
