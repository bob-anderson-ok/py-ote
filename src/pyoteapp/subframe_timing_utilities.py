from typing import Dict, Tuple
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from pyoteapp.genDiffraction import generalizedDiffraction, decide_model_to_use
import numpy as np


def fresnel_length_km(distance_AU: float, wavelength_nm: float = 500.0) -> float:
    """
    Calculates the fresnel length given the wavelength of light and distance to the object.

    :param distance_AU: distance to object in AU (Astronomical Units)
    :param wavelength_nm: wavelength of light in nanometers
    :return: fresnel length in km
    """

    # Convert distance_AU from Astronomical units to km
    distance_km = distance_AU * 149.6e6
    # Convert wavelength_nm to wavelength_km
    wavelength_km = wavelength_nm * 1e-12
    return np.sqrt(distance_km * wavelength_km / 2.0)


def generate_transition_point_time_correction_look_up_tables(
        baseline_intensity: float,
        event_intensity: float,
        frame_time_sec: float,
        asteroid_distance_AU: float = None,
        asteroid_diameter_km: float = None,
        shadow_speed_km_per_sec: float = None,
        centerline_offset=None,
        star_diameter_mas: float = None,
        d_limb_angle_degrees: float = 90.0,
        r_limb_angle_degrees: float = 90.0,
        skip_central_flash: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute D and R lookup tables for finding the time offset from the geometrical
    shadow position of an occultation given a valid transition point

    :param baseline_intensity: B (standard occultation terminology for intensity before occultation)
    :param event_intensity: A (standard occultation terminology for intensity during occultation)
    :param frame_time_sec: light accumulation (integration setting) of camera (seconds)
    :param asteroid_distance_AU: distance to the asteroid/occulting body (in Astronomical Units)
    :param shadow_speed_km_per_sec: speed of the shadow projected at the observing site (km / second)
    :param asteroid_diameter_km: asteroid diameter in km
    :param centerline_offset: distance of observation point from centerline of asteroid path (km)
    :param star_diameter_mas: diameter of star disk (mas - milliarcseconds)
    :param d_limb_angle_degrees: limb angle at disappearance edge (degrees - 90 degrees is head-on)
    :param r_limb_angle_degrees: limb angle at re-appearance edge (degrees - 90 degrees is head-on)
    :param skip_central_flash: set this to True if the (time consuming) cetral flash calculation is not needed
    :return:
    """

    RESOLUTION = 0.0001  # time resolution of lookup tables - 0.1 millisecond

    star_chords_r = None
    star_chords_d = None

    d_graze_values = None
    r_graze_values = None

    if asteroid_distance_AU is None:
        # We cannot calculate a diffraction model lightcurve due to insufficient information,
        # so we revert to an underlying square wave model lightcurve.  In addition, we cannot
        # take into account a finite star disk

        assert star_diameter_mas is None, \
            'Inconsistency: A star diameter was given without the required asteroid distance'

        time_range_seconds = np.ceil(frame_time_sec) + 1.0

        n_points = int(time_range_seconds / RESOLUTION)
        time_values = np.linspace(-time_range_seconds, time_range_seconds, 2 * n_points + 1)
        d_values = np.ndarray(time_values.size)
        r_values = np.ndarray(time_values.size)
        for i, value in enumerate(time_values):
            if value < 0.0:
                r_values[i] = event_intensity
                d_values[i] = baseline_intensity
            else:
                r_values[i] = baseline_intensity
                d_values[i] = event_intensity

        raw_d_values = d_values[:]
        raw_r_values = r_values[:]

        # Prepare the 'sample' that performs a box-car integration when convolved with the model lightcurve

        n_sample_points = round(frame_time_sec / RESOLUTION)
        sample = np.repeat(1.0 / n_sample_points, n_sample_points)

        # Convolve sample against lightcurve to compute the effect of camera frame-time integration.
        d_values = lightcurve_convolve(sample=sample, lightcurve=d_values, shift_needed=len(sample) - 1)
        r_values = lightcurve_convolve(sample=sample, lightcurve=r_values, shift_needed=len(sample) - 1)

        return {'time deltas': time_values, 'D curve': d_values, 'R curve': r_values,
                'star_chords_d': star_chords_d, 'star_chords_r': star_chords_r,
                'raw D': raw_d_values, 'raw R': raw_r_values, 'graze D': d_graze_values, 'graze R': r_graze_values,
                'star D': None, 'star R': None,
                'B': baseline_intensity, 'A': event_intensity}

    elif shadow_speed_km_per_sec is not None:

        # Here we introduce the new diffraction calculation
        x_avg, y_avg, rho, rho_wavelength, _, _, title = \
            generalizedDiffraction(asteroid_diam_km=asteroid_diameter_km,
                                   asteroid_distance_AU=asteroid_distance_AU,
                                   graze_offset_km=centerline_offset,
                                   wavelength1=400, wavelength2=600,
                                   skip_central_flash=skip_central_flash)

        raw_y_avg = np.copy(y_avg)

        y_avg = y_avg * baseline_intensity
        y_avg += event_intensity

        scaled_y_avg = np.copy(y_avg)

        time_values = x_avg / shadow_speed_km_per_sec

        # Prepare the 'sample' that performs a box-car integration when convolved with the model lightcurve
        # to produce the effect of the camera integration.
        n_sample_points = round(frame_time_sec / ((x_avg[1] - x_avg[0]) / shadow_speed_km_per_sec))
        n_sample_points = max(n_sample_points, 1)

        sample = np.repeat(1.0 / n_sample_points, n_sample_points)

        if star_diameter_mas is not None:
            # We are able to compose star chords to convolve with the curves found so far.
            # We have to do that separately for each limb because the limb angle could be different.
            star_chords_d, \
                star_chords_r, \
                star_chords_d_standalone, \
                star_chords_r_standalone = \
                get_star_chord_samples(star_diameter_mas=star_diameter_mas,
                                       distance_to_asteroid_AU=asteroid_distance_AU,
                                       shadow_speed=shadow_speed_km_per_sec,
                                       time_resolution=(x_avg[1] - x_avg[0]) / shadow_speed_km_per_sec,
                                       d_limb_angle_degrees=d_limb_angle_degrees,
                                       r_limb_angle_degrees=r_limb_angle_degrees,
                                       num_points_in_chord_base=len(y_avg)
                                       )
            # print(f'star_chords_d: {star_chords_d}')
            # Convolve sample against lightcurve to compute the effect of star chord integration.
            mid_point = len(y_avg) // 2
            d_values = y_avg[0:mid_point]
            d_values = lightcurve_convolve(sample=star_chords_d_standalone, lightcurve=d_values,
                                           shift_needed=len(star_chords_d_standalone) // 2)
            r_values = y_avg[mid_point: -1]
            r_values = lightcurve_convolve(sample=star_chords_r_standalone, lightcurve=r_values,
                                           shift_needed=len(star_chords_r_standalone) // 2)
            for i, value in enumerate(d_values):
                y_avg[i] = value

            for i, value in enumerate(r_values):
                y_avg[mid_point + i] = value

        # Convolve sample against lightcurve to compute the effect of camera frame-time integration.
        y_avg = lightcurve_convolve(sample=sample, lightcurve=y_avg, shift_needed=len(sample) - 1)

        return {'time deltas': time_values, 'camera response': y_avg,
                'star_chords_d': star_chords_d, 'star_chords_r': star_chords_r,
                'raw underlying': raw_y_avg, 'scaled underlying': scaled_y_avg,
                'B': baseline_intensity, 'A': event_intensity}


def intensity_at_time(data, time, edge_type):
    assert edge_type in ['D', 'R']

    if edge_type == 'D':
        if time <= data['time deltas'][0]:
            return data['D curve'][0]
        if time >= data['time deltas'][-1]:
            return data['D curve'][-1]
        for i, t in enumerate(data['time deltas']):
            if t >= time:
                return data['D curve'][i]
        return None  # This should never happen

    else:
        if time < data['time deltas'][0]:
            return data['R curve'][0]
        if time > data['time deltas'][-1]:
            return data['R curve'][-1]
        for i, t in enumerate(data['time deltas']):
            if t >= time:
                return data['R curve'][i]
        return None  # This should never happen


def lightcurve_convolve(sample: np.ndarray, lightcurve: np.ndarray, shift_needed: int) -> np.ndarray:
    """
    Computes the convolution of sample[] against lightcurve[] with corrections for leading edge
    effects and for the right shift of the normal convolution calculation.   We do a
    counter-shift so as to maintain our start-of-exposure convention.

    :param sample: usually contains either a 'box-car' (for frame integration) or a set of star chords
    :param lightcurve: the underlying lightcurve (a square wave or a diffraction curve)
    :param shift_needed: size of left shift desired
    :return: convolution of sample against lightcurve (with length = len(lightcurve)
    """

    # To 'eliminate' the leading edge effect inherent in np.convolve(), we insert a compensating number of
    # duplicate points at the beginning.  This works well only if the lightcurve has already achieved
    # 'steady state' in this region.  We strive to guarantee this everywhere this function is called.

    leading_edge_extension = np.ones(len(sample) - 1) * lightcurve[0]
    lightcurve_extended = np.concatenate((leading_edge_extension, lightcurve))

    new_lightcurve = np.convolve(sample, lightcurve_extended, 'valid')
    assert len(new_lightcurve) == len(lightcurve)

    # Now perform a left shift to make our convolution consistent with our start-of-exposure convention.
    # shift_count = len(sample) - 1
    shift_count = shift_needed
    lightcurve_shifted = np.roll(new_lightcurve, -shift_count)  # Do a 'left-roll'
    # Fix the points at the right edge that got overwritten by the 'roll'  Here we assume that the
    # lightcurve had already achieved 'steady state' at the end.
    for i in range(1, shift_count + 1):
        lightcurve_shifted[-i] = new_lightcurve[-1]

    return lightcurve_shifted


def sin_degrees(angle_degrees):
    # Convert angle from degrees to radians
    radians_per_degree = np.pi / 180
    angle_radians = angle_degrees * radians_per_degree
    return np.sin(angle_radians)


def get_star_chord_samples(
        star_diameter_mas: float,
        distance_to_asteroid_AU: float,
        shadow_speed: float,
        time_resolution: float,
        d_limb_angle_degrees: float = 90.0,
        r_limb_angle_degrees: float = 90.0,
        num_points_in_chord_base: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the array that can be convolved with the diffraction curve to incorporate star diameter effects.

    :param star_diameter_mas:        star disk diameter (units: milli-arcseconds)
    :param distance_to_asteroid_AU:  distance to asteroid (units: Astronomical Units)
    :param shadow_speed: asteroid speed (units: km/sec)
    :param time_resolution: resolution for chord slices (units: sec)
    :param d_limb_angle_degrees: limb angle at D (90 degrees = head-on)
    :param r_limb_angle_degrees: limb angle at R (90 degrees = head-on)
    :param num_points_in_chord_base: number of points in x axis of main plot
    :return: np.array of chord profiles, normalized so that sum(chords) = 1.0
    """
    star_diameter_radians = star_diameter_mas * 4.84814e-9
    distance_to_asteroid_km = distance_to_asteroid_AU * 149.6e6

    # print(star_diameter_radians, np.tan(star_diameter_radians), np.sin(star_diameter_radians))
    star_projection_km = 2 * np.tan(star_diameter_radians / 2) * distance_to_asteroid_km
    star_diameter_sec = star_projection_km / shadow_speed

    n_star_chords_in_radius = int(star_diameter_sec / 2 / time_resolution)
    print(f'time_resolution: {time_resolution:0.4f}  n_star_chords: {n_star_chords_in_radius}')
    n_r_limb_chords_in_radius = int(n_star_chords_in_radius / sin_degrees(r_limb_angle_degrees))
    n_d_limb_chords_in_radius = int(n_star_chords_in_radius / sin_degrees(d_limb_angle_degrees))

    print(f'n_d_limb_chords_in_radius: {n_d_limb_chords_in_radius}  '
          f'n_r_limb_chords_in_radius: {n_r_limb_chords_in_radius}')

    d_time_resolution = time_resolution * sin_degrees(d_limb_angle_degrees)
    r_time_resolution = time_resolution * sin_degrees(r_limb_angle_degrees)

    radius_sec = star_diameter_sec / 2
    r2 = radius_sec * radius_sec

    print(f'radius_sec**2: {r2:0.4f}')

    d_star_chords = np.zeros(num_points_in_chord_base)
    d_star_chords_standalone = np.zeros(n_d_limb_chords_in_radius * 2 + 1)
    normalizer = 0.0
    plot_margin = 20
    j = 0
    for i in range(-n_d_limb_chords_in_radius, n_d_limb_chords_in_radius + 1):
        chord = 2.0 * np.sqrt(r2 - (i * d_time_resolution) ** 2)
        if j == 0 or j == n_d_limb_chords_in_radius * 2:
            chord = 0
        d_star_chords[j + plot_margin] = chord
        d_star_chords_standalone[j] = chord
        j += 1
        normalizer += chord

    normed_d_chords = [c / normalizer for c in d_star_chords]
    normed_d_chords_standalone = [c / normalizer for c in d_star_chords_standalone]

    r_star_chords = np.zeros(num_points_in_chord_base)
    r_star_chords_standalone = np.zeros(n_r_limb_chords_in_radius * 2 + 1)

    normalizer = 0.0
    plot_offset = len(r_star_chords) - n_r_limb_chords_in_radius * 2 - plot_margin
    j = 0
    for i in range(-n_r_limb_chords_in_radius, n_r_limb_chords_in_radius + 1):
        chord = 2.0 * np.sqrt(r2 - (i * r_time_resolution) ** 2)
        if j == 0 or j == n_r_limb_chords_in_radius * 2:
            chord = 0.0
        r_star_chords[j + plot_offset] = chord
        r_star_chords_standalone[j] = chord
        j += 1
        normalizer += chord

    normed_r_chords = [c / normalizer for c in r_star_chords]
    normed_r_chords_standalone = [c / normalizer for c in r_star_chords_standalone]

    return np.array(normed_d_chords), np.array(normed_r_chords), \
        np.array(normed_d_chords_standalone), np.array(normed_r_chords_standalone)


def generate_underlying_lightcurve_plots(
        b_value=100.0,
        a_value=0.0,
        frame_time=None,
        ast_dist=None,
        shadow_speed=None,
        ast_diam=None,
        centerline_offset=None,
        star_diam=None,
        d_angle=None,
        r_angle=None,
        skip_central_flash=False,
        title_addon=''
):
    mid = (b_value + a_value) / 2

    if frame_time > 0.001:
        data_summary = f'\nframe time(sec): {frame_time:0.4f} '
    else:
        data_summary = '\n'
    if ast_dist is not None:
        data_summary += f'  asteroid distance(AU): {ast_dist:0.2f}'
    if shadow_speed is not None:
        data_summary += f'  shadow speed(km/sec): {shadow_speed:0.2f}'
    extra_title = title_addon

    ans = generate_transition_point_time_correction_look_up_tables(
        baseline_intensity=b_value,
        event_intensity=a_value,
        frame_time_sec=frame_time,
        asteroid_distance_AU=ast_dist,
        asteroid_diameter_km=ast_diam,
        shadow_speed_km_per_sec=shadow_speed,
        centerline_offset=centerline_offset,
        star_diameter_mas=star_diam,
        d_limb_angle_degrees=d_angle,
        r_limb_angle_degrees=r_angle,
        skip_central_flash=skip_central_flash
    )
    fig = plt.figure('Dplot', figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.set(xlabel='seconds', ylabel='Intensity')
    if star_diam is not None:
        star_comment = f'\nstar diam(mas): {star_diam:0.2f}  D limb angle: {d_angle:0.1f}' \
                       f'  R limb angle: {r_angle:0.1f}'
    else:
        star_comment = ''
    if ast_diam is not None and centerline_offset is not None:
        graze_comment = f'\nast diam(km): {ast_diam:0.2f} centerline offset(km): {centerline_offset:0.2f}'
    else:
        graze_comment = ''
    ax.set_title(extra_title + 'Lightcurve composition' + data_summary + star_comment + graze_comment)
    if ans['star_chords_d'] is not None:
        star_chords_d = ans['star_chords_d']
        rescaled_star_chords_d = star_chords_d * (b_value - a_value) / max(star_chords_d) / 2
        rescaled_star_chords_d += a_value
        n_star_chords = len(rescaled_star_chords_d)
        ax.plot(ans['time deltas'][:n_star_chords],
                rescaled_star_chords_d, label='star disk function', color='orange')

        star_chords_r = ans['star_chords_r']
        rescaled_star_chords_r = star_chords_r * (b_value - a_value) / max(star_chords_r) / 2
        rescaled_star_chords_r += a_value
        n_star_chords = len(rescaled_star_chords_r)
        ax.plot(ans['time deltas'][:n_star_chords],
                rescaled_star_chords_r, color='orange')

    right_edge_shadow = (ast_diam / 2) / shadow_speed
    ax.axvline(right_edge_shadow, linestyle='--', label='geometrical shadow', color='black')
    ax.axvline(-right_edge_shadow, linestyle='--', color='black')
    if frame_time > 0.001:
        offset = ans['time deltas'][-1] / 2
        ax.plot([offset, offset, offset + frame_time, offset + frame_time],
                [a_value, mid, mid, a_value], label='camera exposure function', color='red')
    # if ans['graze D'] is not None:
    #     ax.plot(ans['time deltas'], ans['graze D'], label='diffraction (grazed) D')
    # if ans['star D'] is None:
    #     ax.plot(ans['time deltas'], ans['raw D'], label='underlying lightcurve', color='green')
    # else:
    ax.plot(ans['time deltas'], ans['scaled underlying'], label='underlying lightcurve', color='green')

    if frame_time > 0.001:
        ax.plot(ans['time deltas'], ans['camera response'], label='camera response', color='red')

    plt.grid()
    ax.legend()
    main_fig = fig

    return main_fig, ans


def demo():
    main_figure = None

    tests_to_run = [3]
    print(f'=== tests to be run: {tests_to_run}')

    if 1 in tests_to_run:
        print(f'running test 1')
        lightcurve = np.concatenate((np.zeros(3), np.ones(2), np.zeros(2)))
        sample = np.ones(2)
        ans = lightcurve_convolve(sample=sample, lightcurve=lightcurve, shift_needed=len(sample) - 1)
        assert np.array_equal(ans, np.array([0, 0, 1, 2, 1, 0, 0]))

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        ax.plot(ans, 'v', label='convolution')
        ax.plot(lightcurve, '^', label='lightcurve')
        ax.plot(sample, 'x', label='sampler')
        ax.legend()
        plt.show()

    if 2 in tests_to_run:
        print(f'running test 2')
        ans = generate_transition_point_time_correction_look_up_tables(
            baseline_intensity=10.0,
            event_intensity=-1.5, frame_time_sec=0.334
        )

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(ans['time deltas'], ans['D curve'], '.', label='D curve')
        ax.legend()

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(ans['time deltas'], ans['R curve'], '.', label='R curve')
        ax.legend()

    if 3 in tests_to_run:
        print(f'running test 3')
        b_value = 100.0
        a_value = 0.0
        skip_central_flash = True  # This speeds up computation when the central flash is not needed
        # frame_time = 0.001 # Will show underlying diffraction curve
        frame_time = 0.08
        # frame_time = 0.001
        star_diam = 14.2  # mas
        # star_diam = None
        d_angle = 75
        r_angle = 90
        # ast_dist = 2.5752  # Felicia
        ast_dist = 1.978
        ast_diam = 45
        centerline_offset = 0.0
        shadow_speed = 6.49

        decide_model_to_use(asteroid_diameter_km=ast_diam,
                            asteroid_distance_AU=ast_dist,
                            star_diameter_mas=star_diam,
                            shadow_speed=shadow_speed,
                            frame_time=frame_time)

        title_addon = '(Roma)  '

        main_figure, _ = generate_underlying_lightcurve_plots(
            b_value=100.0,
            a_value=0.0,
            frame_time=frame_time,
            ast_dist=ast_dist,
            ast_diam=ast_diam,
            centerline_offset=centerline_offset,
            shadow_speed=shadow_speed,
            star_diam=star_diam,
            d_angle=d_angle,
            r_angle=r_angle,
            title_addon=title_addon,
            skip_central_flash=skip_central_flash
        )

        _ = generate_transition_point_time_correction_look_up_tables(
            baseline_intensity=b_value,
            event_intensity=a_value,
            frame_time_sec=frame_time,
            asteroid_distance_AU=ast_dist,
            asteroid_diameter_km=ast_diam,
            centerline_offset=centerline_offset,
            shadow_speed_km_per_sec=shadow_speed,
            star_diameter_mas=star_diam,
            d_limb_angle_degrees=d_angle,
            r_limb_angle_degrees=r_angle,
            skip_central_flash=skip_central_flash
        )

    print(f'=== end tests')

    return main_figure


if __name__ == "__main__":
    # print(plt.get_backend())
    # plt.switch_backend('Qt5agg')
    # print(plt.get_backend())
    main_plot = demo()
    matplotlib.pyplot.show()
