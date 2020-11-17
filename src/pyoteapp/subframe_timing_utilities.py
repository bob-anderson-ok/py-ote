from typing import Dict, Tuple
import numpy as np
import matplotlib
import pickle
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy import interpolate


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
        baseline_intensity: float, event_intensity: float, frame_time_sec: float,
        asteroid_distance_AU: float = None,
        shadow_speed_km_per_sec: float = None,
        ast_diam=None,
        centerline_offset=None,
        star_diameter_mas: float = None,
        d_limb_angle_degrees: float = 90.0,
        r_limb_angle_degrees: float = 90.0,
        suppress_diffraction: bool = True,
        diff_table_path=''
) -> Dict[str, np.ndarray]:
    """
    Compute D and R lookup tables for finding the time offset from the geometrical
    shadow position of an occultation given a valid transition point

    :param baseline_intensity: B (standard occultation terminology for intensity before occultation)
    :param event_intensity: A (standard occultation terminology for intensity during occultation)
    :param frame_time_sec: light accumulation (integration setting) of camera (seconds)
    :param asteroid_distance_AU: distance to the asteroid/occulting body (in Astronomical Units)
    :param shadow_speed_km_per_sec: speed of the shadow projected at the observing site (km / second)
    :param ast_diam: asteroid diameter in km
    :param centerline_offset: distance of observation point from centerline of asteroid path (km)
    :param star_diameter_mas: diameter of star disk (mas - milliarcseconds)
    :param d_limb_angle_degrees: limb angle at disappearance edge (degrees - 90 degrees is head-on)
    :param r_limb_angle_degrees: limb angle at re-appearance edge (degrees - 90 degrees is head-on)
    :param suppress_diffraction: set this Fale if you want to see diffraction effect
    :param diff_table_path: path to generic diffraction table
    :return:
    """

    RESOLUTION = 0.0001   # time resolution of lookup tables - 0.1 millisecond

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
        # This code group utilizes a pre-computed integrated diffraction curve. We know that both asteroid_distance_AU
        # and shadow_speed_km_per_sec are both available.
        fresnel_length = fresnel_length_km(distance_AU=asteroid_distance_AU)
        fresnel_unit_time = fresnel_length / shadow_speed_km_per_sec
        time_for_10_fresnel_units = 10.0 * fresnel_unit_time

        # Fetch the pre-computed integrated (multi-wavelength) diffraction curve
        pickle_file = open(diff_table_path, 'rb')
        table = pickle.load(pickle_file)
        u_values = table['u']
        d_values = table['D'] * (baseline_intensity - event_intensity)
        r_values = table['R'] * (baseline_intensity - event_intensity)
        d_values += event_intensity
        r_values += event_intensity

        if suppress_diffraction:
            for i in range(d_values.size):
                if u_values[i] <= 0.0:
                    d_values[i] = baseline_intensity
                else:
                    d_values[i] = event_intensity

            for i in range(r_values.size):
                if u_values[i] <= 0.0:
                    r_values[i] = event_intensity
                else:
                    r_values[i] = baseline_intensity

        if star_diameter_mas is None:
            time_needed_for_good_curve = 4.0 * frame_time_sec
        else:
            # We have to compute the time needed for the star projection to pass, using the
            # limb angle that is smallest.
            min_limb_angle_degrees = min(d_limb_angle_degrees, r_limb_angle_degrees)
            star_diameter_radians = star_diameter_mas * 4.84814e-9
            distance_to_asteroid_km = asteroid_distance_AU * 149.6e6
            # print(star_diameter_radians, np.tan(star_diameter_radians), np.sin(star_diameter_radians))
            star_projection_km = np.tan(star_diameter_radians) * distance_to_asteroid_km
            star_projection_time_sec = star_projection_km / \
                shadow_speed_km_per_sec / sin_degrees(min_limb_angle_degrees)
            # print(f'frame_time: {frame_time_sec}   star_time: {star_projection_time_sec}')
            if star_projection_time_sec > frame_time_sec:
                time_needed_for_good_curve = 4.0 * star_projection_time_sec
            else:
                time_needed_for_good_curve = 4.0 * frame_time_sec

        if time_for_10_fresnel_units < time_needed_for_good_curve:
            # We need to extend the arrays loaded from the pickle_file
            time_extension_needed = time_needed_for_good_curve - time_for_10_fresnel_units
            extended_curves = time_extend_lightcurves(
                time_extension_needed, fresnel_unit_time, u_values, d_values, r_values
            )
            u_values = extended_curves['u_values']
            d_values = extended_curves['d_values']
            r_values = extended_curves['r_values']

        raw_d_values = np.copy(d_values)
        raw_r_values = np.copy(r_values)

        if ast_diam is not None and centerline_offset is not None:
            d_graze_values = np.ndarray([len(u_values)])
            r_graze_values = np.ndarray([len(u_values)])
            # We need to adjust the diffraction light curves for a possible
            # off centerline observation. First we create two interpolation functions:
            d_interp_func = interpolate.interp1d(
                u_values, d_values, kind='quadratic',
                bounds_error=False, fill_value=(d_values[0], d_values[-1]))
            r_interp_func = interpolate.interp1d(
                u_values, r_values, kind='quadratic',
                bounds_error=False, fill_value=(r_values[0], r_values[-1]))

            r_ast = ast_diam / 2.0 / fresnel_length
            g = centerline_offset / fresnel_length
            for i in range(len(u_values)):
                r = np.sqrt(r_ast ** 2 + u_values[i] ** 2 + 2 * r_ast * np.abs(u_values[i]) * np.sqrt(
                    1.0 - (g ** 2 / r_ast ** 2)))
                d_graze_values[i] = d_interp_func(np.sign(u_values[i]) * (r - r_ast))
                r_graze_values[i] = r_interp_func(np.sign(u_values[i]) * (r - r_ast))
            d_values = d_graze_values
            r_values = r_graze_values

        time_values = u_values * fresnel_unit_time

        # Prepare the 'sample' that performs a box-car integration when convolved with the model lightcurve
        # to produce the effect of the camera integration. We have to convert time values to u values for
        # this calculation.
        n_sample_points = round(frame_time_sec / fresnel_unit_time / (u_values[1] - u_values[0]))
        n_sample_points = max(n_sample_points, 1)
        sample = np.repeat(1.0 / n_sample_points, n_sample_points)

        # print(f'n_sample: {len(sample)}  n_lightcurve: {len(d_values)}')

        star_d_values = None
        star_r_values = None

        if star_diameter_mas is not None:
            # We are able to compose star chords to convolve with the curves found so far.
            # We have to do that separately for each limb because the limb angle could be different.
            star_chords_d, star_chords_r = get_star_chord_samples(
                star_diameter_mas, asteroid_distance_AU,
                fresnel_length, u_values[1] - u_values[0], d_limb_angle_degrees, r_limb_angle_degrees)
            # Convolve sample against lightcurve to compute the effect of star chord integration.
            d_values = lightcurve_convolve(sample=star_chords_d, lightcurve=d_values,
                                           shift_needed=len(star_chords_d) // 2)
            r_values = lightcurve_convolve(sample=star_chords_r, lightcurve=r_values,
                                           shift_needed=len(star_chords_r) // 2)
            star_d_values = np.copy(d_values)
            star_r_values = np.copy(r_values)

        # Convolve sample against lightcurve to compute the effect of camera frame-time integration.
        d_values = lightcurve_convolve(sample=sample, lightcurve=d_values, shift_needed=len(sample) - 1)
        r_values = lightcurve_convolve(sample=sample, lightcurve=r_values, shift_needed=len(sample) - 1)

        return {'time deltas': time_values, 'D curve': d_values, 'R curve': r_values,
                'star_chords_d': star_chords_d, 'star_chords_r': star_chords_r,
                'raw D': raw_d_values, 'raw R': raw_r_values, 'graze D': d_graze_values, 'graze R': r_graze_values,
                'star D': star_d_values, 'star R': star_r_values, 'B': baseline_intensity, 'A': event_intensity}


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


def time_extend_lightcurves(time_extension, fresnel_unit_time, u_values, d_values, r_values):
    fresnel_extension_needed = time_extension / fresnel_unit_time
    n_increments = int(100 * round(fresnel_extension_needed / 2.0))
    # print(f'n_increments type: {type(n_increments)}')
    delta_u = u_values[1] - u_values[0]
    left_u_ext = np.linspace(-(n_increments + 1) * delta_u, -delta_u, num=n_increments + 1)
    left_u_ext += u_values[0]
    right_u_ext = np.linspace(delta_u, (n_increments + 1) * delta_u, num=n_increments + 1)
    right_u_ext += u_values[-1]
    left_d_ext = np.repeat(d_values[0], n_increments + 1)
    left_r_ext = np.repeat(r_values[0], n_increments + 1)
    right_d_ext = np.repeat(d_values[-1], n_increments + 1)
    right_r_ext = np.repeat(r_values[-1], n_increments + 1)
    extended_u_values = np.concatenate((left_u_ext, u_values, right_u_ext))
    extended_d_values = np.concatenate((left_d_ext, d_values, right_d_ext))
    extended_r_values = np.concatenate((left_r_ext, r_values, right_r_ext))
    return {'u_values': extended_u_values, 'd_values': extended_d_values, 'r_values': extended_r_values}


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
    lightcurve_shifted = np.roll(new_lightcurve, -shift_count)   # Do a 'left-roll'
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
        delta_u: float,
        fresnel_length: float, d_limb_angle_degrees: float = 90.0, r_limb_angle_degrees: float = 90.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the array that can be convolved with the diffraction curve to incorporate star diameter effects.

    :param star_diameter_mas:        star disk diameter (units: milli-arcseconds)
    :param distance_to_asteroid_AU:  distance to asteroid (units: Astronomical Units)
    :param fresnel_length: fresnel scale (units: km)
    :param delta_u: resolution of u_values (units: fresnel numbers)
    :param d_limb_angle_degrees: limb angle at D (90 degrees = head-on)
    :param r_limb_angle_degrees: limb angle at R (90 degrees = head-on)
    :return: np.array of chord profiles, normalized so that sum(chords) = 1.0
    """
    star_diameter_radians = star_diameter_mas * 4.84814e-9
    distance_to_asteroid_km = distance_to_asteroid_AU * 149.6e6

    # print(star_diameter_radians, np.tan(star_diameter_radians), np.sin(star_diameter_radians))
    star_projection_km = np.tan(star_diameter_radians) * distance_to_asteroid_km
    star_diameter_u = star_projection_km / fresnel_length

    n_star_chords = int(star_diameter_u / delta_u / 2)
    n_r_limb_chords = int(n_star_chords / sin_degrees(r_limb_angle_degrees))
    n_d_limb_chords = int(n_star_chords / sin_degrees(d_limb_angle_degrees))

    # print(f'n_d_limb_chords: {n_d_limb_chords}  n_r_limb_chords: {n_r_limb_chords}')

    radius_u = star_diameter_u / 2
    r2 = radius_u * radius_u

    d_star_chords = []
    delta_u_d = delta_u * sin_degrees(d_limb_angle_degrees)
    normalizer = 0.0
    for i in range(-n_d_limb_chords, n_d_limb_chords + 1):
        chord = 2.0 * np.sqrt(r2 - (i * delta_u_d)**2)
        d_star_chords.append(chord)
        normalizer += chord

    normed_d_chords = [c / normalizer for c in d_star_chords]

    r_star_chords = []
    delta_u_r = delta_u * sin_degrees(r_limb_angle_degrees)
    normalizer = 0.0
    for i in range(-n_r_limb_chords, n_r_limb_chords + 1):
        chord = 2.0 * np.sqrt(r2 - (i * delta_u_r) ** 2)
        r_star_chords.append(chord)
        normalizer += chord

    normed_r_chords = [c / normalizer for c in r_star_chords]

    return np.array(normed_d_chords), np.array(normed_r_chords)


def time_correction(correction_dict, transition_point_intensity, edge_type='D'):
    # assert correction_dict['A'] <= transition_point_intensity <= correction_dict['B']
    if not correction_dict['A'] <= transition_point_intensity <= correction_dict['B']:
        print('Intensity violation encountered: ', correction_dict['A'], transition_point_intensity, correction_dict['B'])
    assert edge_type == 'D' or edge_type == 'R'

    # We start our search from the middle and work either up or down to find the best matching intensity.
    # We return the negative of the corresponding time_delta as the needed time correction
    middle_intensity = (correction_dict['B'] + correction_dict['A']) / 2

    if edge_type == 'D':
        curve_to_use = correction_dict['D curve']
        # Find the index of the middle value in the intensity table
        mid_index = np.where(curve_to_use <= middle_intensity)[0][0]
        # print(mid_index)
        if transition_point_intensity >= curve_to_use[mid_index]:
            # We need to search to the left
            search_index = mid_index
            while search_index > 0:
                if transition_point_intensity <= curve_to_use[search_index]:
                    return -correction_dict['time deltas'][search_index]
                search_index -= 1
            return None  # This return should NEVER be reached
        else:
            # We need to search to the right
            search_index = mid_index
            while search_index < curve_to_use.size:
                if transition_point_intensity >= curve_to_use[search_index]:
                    return -correction_dict['time deltas'][search_index]
                search_index += 1
            return None  # This return should NEVER be reached
    else:
        curve_to_use = correction_dict['R curve']
        # Find the index of the middle value in the intensity table
        mid_index = np.where(curve_to_use >= middle_intensity)[0][0]
        # print(mid_index)
        if transition_point_intensity >= curve_to_use[mid_index]:
            # We need to search to the right
            search_index = mid_index
            while search_index < curve_to_use.size:
                if transition_point_intensity <= curve_to_use[search_index]:
                    return -correction_dict['time deltas'][search_index]
                search_index += 1
            return None  # This return should NEVER be reached
        else:
            # We need to search to the left
            search_index = mid_index
            while search_index > 0:
                if transition_point_intensity >= curve_to_use[search_index]:
                    return -correction_dict['time deltas'][search_index]
                search_index -= 1
            return None  # This return should NEVER be reached


def generate_underlying_lightcurve_plots(
        diff_table_path='',
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
        suppress_diffraction=True,
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
        shadow_speed_km_per_sec=shadow_speed,
        ast_diam=ast_diam,
        centerline_offset=centerline_offset,
        star_diameter_mas=star_diam,
        d_limb_angle_degrees=d_angle,
        r_limb_angle_degrees=r_angle,
        suppress_diffraction=suppress_diffraction,
        diff_table_path=diff_table_path
    )
    fig = plt.figure('Dplot', figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.set(xlabel='seconds', ylabel='Intensity')
    if star_diam is not None:
        star_comment = f'\nstar diam(mas): {star_diam:0.2f}  limb angle: {d_angle:0.1f}'
    else:
        star_comment = ''
    if ast_diam is not None and centerline_offset is not None:
        graze_comment = f'\nast diam(km): {ast_diam:0.2f} centerline offset(km): {centerline_offset:0.2f}'
    else:
        graze_comment = ''
    ax.set_title(extra_title + 'D underlying lightcurve info' + data_summary + star_comment + graze_comment)
    if frame_time > 0.001:
        ax.plot(ans['time deltas'], ans['D curve'], label='camera response')
    if ans['star_chords_d'] is not None:
        star_chords_d = ans['star_chords_d']
        star_chords_d[0] = 0.0
        star_chords_d[-1] = 0.0
        rescaled_star_chords_d = star_chords_d * (b_value - a_value) / max(star_chords_d) / 2
        rescaled_star_chords_d += a_value
        n_star_chords = len(rescaled_star_chords_d)
        ax.plot(ans['time deltas'][:n_star_chords], rescaled_star_chords_d, label='star disk function')
    ax.axvline(0.0, linestyle='--', label='geometrical shadow')
    if frame_time > 0.001:
        offset = ans['time deltas'][-1] / 2
        ax.plot([offset, offset, offset + frame_time, offset + frame_time],
                [a_value, mid, mid, a_value], label='camera exposure function')
    if ans['graze D'] is not None:
        ax.plot(ans['time deltas'], ans['graze D'], label='graze D')
    if ans['star D'] is None:
        ax.plot(ans['time deltas'], ans['raw D'], label='underlying lightcurve')
    else:
        ax.plot(ans['time deltas'], ans['star D'], label='underlying lightcurve')

    plt.grid()
    ax.legend()
    d_fig = fig

    fig = plt.figure('Rplot', figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.set(xlabel='seconds', ylabel='Intensity')
    if star_diam is not None:
        star_comment = f'\nstar diam(mas): {star_diam:0.2f}  limb angle: {r_angle:0.1f}'
    else:
        star_comment = ''
    ax.set_title(extra_title + 'R underlying lightcurve info' + data_summary + star_comment + graze_comment)
    if frame_time > 0.001:
        ax.plot(ans['time deltas'], ans['R curve'], label='camera response')
    if ans['star_chords_r'] is not None:
        star_chords_r = ans['star_chords_r']
        star_chords_r[0] = 0.0
        star_chords_r[-1] = 0.0
        rescaled_star_chords_r = star_chords_r * (b_value - a_value) / max(star_chords_r) / 2
        rescaled_star_chords_r += a_value
        n_star_chords = len(rescaled_star_chords_r)
        ax.plot(ans['time deltas'][:n_star_chords], rescaled_star_chords_r, label='star disk function')
    ax.axvline(0.0, linestyle='--', label='geometrical shadow')
    ax.axhline(a_value, linestyle='dotted', label='baseline intensity')
    if frame_time > 0.001:
        offset = ans['time deltas'][-1] / 2
        ax.plot([offset, offset, offset + frame_time, offset + frame_time],
                [a_value, mid, mid, a_value], label='camera exposure function')
    if ans['graze R'] is not None:
        ax.plot(ans['time deltas'], ans['graze R'], label='graze R')
    if ans['star R'] is None:
        ax.plot(ans['time deltas'], ans['raw R'], label='underlying lightcurve')
    else:
        ax.plot(ans['time deltas'], ans['star R'], label='underlying lightcurve')

    plt.grid()
    ax.legend()
    r_fig = fig

    return d_fig, r_fig, ans


def demo(diff_table_path):
    d_figure = None
    r_figure = None

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
        # frame_time = 0.001 # Will show underlying diffraction curve
        frame_time = 0.0334 * 4
        star_diam = 0.5  # mas
        # star_diam = None
        d_angle = 30
        r_angle = 90
        ast_dist = 2.5752     # Felicia
        shadow_speed = 4.55   # Felicia
        title_addon = '(Felicia 01062020 Watec)  '

        d_figure, r_figure, _ = generate_underlying_lightcurve_plots(
            diff_table_path=diff_table_path,
            b_value=100.0,
            a_value=0.0,
            frame_time=frame_time,
            ast_dist=ast_dist,
            shadow_speed=shadow_speed,
            star_diam=star_diam,
            d_angle=d_angle,
            r_angle=r_angle,
            title_addon=title_addon
        )

        ans = generate_transition_point_time_correction_look_up_tables(
            baseline_intensity=b_value,
            event_intensity=a_value,
            frame_time_sec=frame_time,
            asteroid_distance_AU=ast_dist,
            shadow_speed_km_per_sec=shadow_speed,
            star_diameter_mas=star_diam,
            d_limb_angle_degrees=d_angle,
            r_limb_angle_degrees=r_angle,
            diff_table_path=diff_table_path
        )

        time_adjustment = time_correction(ans, 80, 'D')
        print(f'D time_adjustment @ 80: {time_adjustment}')
        time_adjustment = time_correction(ans, 20, 'D')
        print(f'D time_adjustment @ 20: {time_adjustment}')
        time_adjustment = time_correction(ans, 80, 'R')
        print(f'R time_adjustment @ 80: {time_adjustment}')
        time_adjustment = time_correction(ans, 20, 'R')
        print(f'R time_adjustment @ 20: {time_adjustment}')

    print(f'=== end tests')

    return d_figure, r_figure


if __name__ == "__main__":
    # print(plt.get_backend())
    # plt.switch_backend('Qt5agg')
    # print(plt.get_backend())
    d_plot, r_plot = demo(diff_table_path='diffraction-table.p')
    matplotlib.pyplot.show()
