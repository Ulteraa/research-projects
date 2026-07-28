from calibgraph.monitoring.calibration_health import (
    fit_calibration_health_monitor,
)
from calibgraph.simulation.drift import (
    generate_multicamera_drift_sequence,
)


def test_large_drift_is_detected_and_localized():
    sequence = generate_multicamera_drift_sequence(
        num_steps=60,
        calibration_window=15,
        drift_start_index=35,
        drift_camera="wrist_camera",
        drift_translation_mm=8.0,
        drift_rotation_deg=1.5,
        seed=12,
    )
    monitor = fit_calibration_health_monitor(
        sequence.dataset,
        calibration_window=sequence.calibration_window,
    )
    report = monitor.evaluate(sequence.dataset)

    valid_detections = [
        detection
        for detection in (
            report.first_detection_by_camera.values()
        )
        if detection is not None and detection >= 35
    ]
    assert valid_detections
    system_detection = min(valid_detections)
    assert system_detection <= 40
    assert (
        report.suspected_camera_by_time[system_detection]
        == "wrist_camera"
    )


def test_no_drift_does_not_trigger_recalibration():
    sequence = generate_multicamera_drift_sequence(
        num_steps=60,
        calibration_window=15,
        drift_start_index=None,
        drift_camera=None,
        drift_translation_mm=0.0,
        drift_rotation_deg=0.0,
        seed=13,
    )
    monitor = fit_calibration_health_monitor(
        sequence.dataset,
        calibration_window=sequence.calibration_window,
    )
    report = monitor.evaluate(sequence.dataset)

    assert all(
        detection is None
        for detection in (
            report.first_detection_by_camera.values()
        )
    )
