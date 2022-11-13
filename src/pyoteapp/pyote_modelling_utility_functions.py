import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
import math

from numpy import zeros, complex128
from numba import jit, prange
import matplotlib.pyplot as plt
from scipy.fftpack import fft2
from numpy import exp, pi
from scipy.special import fresnel
from matplotlib import patches
from scipy import interpolate
# from pathlib import Path
# from datetime import datetime, timezone


def get_star_chord_samples(x, plot_margin, LCP) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the array that can be convolved with the diffraction curve to incorporate star diameter effects.

    :param x: x values of underlying lightcurve
    :param plot_margin: offset the plot so that it looks better
    :param LCP: lightcurve parameters
    :return: np.array of chord profiles, normalized so that sum(chords) = 1.0
    """
    # star_diameter_radians = LCP.star_diameter_mas * 4.84814e-9
    # distance_to_asteroid_km = LCP.asteroid_distance_AU * 149.6e6

    # star_diameter_sec = LCP.star_diameter_km / LCP.shadow_speed

    distance_resolution = x[1] - x[0]

    n_star_chords_in_radius = int(LCP.star_diameter_km / 2 / distance_resolution)

    # print(f'distance_resolution: {distance_resolution:0.4f}  n_star_chords: {n_star_chords_in_radius}')
    # print(f'n_star_chords_in_radius: {n_star_chords_in_radius}')

    r2 = LCP.star_radius_km ** 2

    star_chords = np.zeros(len(x))
    star_chords_standalone = np.zeros(n_star_chords_in_radius * 2 + 1)
    normalizer = 0.0
    plot_offset = len(x) - plot_margin - 2 * n_star_chords_in_radius
    j = 0
    max_chord = None
    for i in range(-n_star_chords_in_radius, n_star_chords_in_radius + 1):
        chord = 2.0 * np.sqrt(r2 - (i * distance_resolution) ** 2)
        if j == 0 or j == n_star_chords_in_radius * 2:
            chord = 0
        star_chords[j + plot_offset] = chord
        star_chords_standalone[j] = chord
        j += 1
        normalizer += chord
        if i == 0:
            max_chord = chord

    normed_chords_standalone = [c / normalizer for c in star_chords_standalone]

    scaler = max_chord * 2
    normed_chords = [c / scaler for c in star_chords]

    return np.array(normed_chords), np.array(normed_chords_standalone)


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


def scaleToADU(y_values, LCP):
    # Scale y_values to ADU (modify in place)

    # Adjust for a possible graze (in a graze, min(y) is greater than zero)
    # minY = np.min(y_values)
    # y_values -= minY
    # y_values *= (1.0 / (1.0 - minY))

    # Now scale to ADU
    # y_values *= (LCP.baseline_ADU - LCP.bottom_ADU)
    # y_values += LCP.bottom_ADU

    compression = 1.0 - LCP.bottom_ADU / LCP.baseline_ADU
    y_values = ((y_values - 1.0) * compression + 1.0) * LCP.baseline_ADU
    # y_values *= LCP.baseline_ADU

    return y_values


@jit(nopython=True)
def turnDir(ax, ay, bx, by, cx, cy) -> int:
    # ab = Vector(b.x - a.x, b.y - a.y)
    # ac = Vector(c.x - a.x, c.y - a.y)

    abx = bx - ax
    aby = by - ay
    acx = cx - ax
    acy = cy - ay

    cross_product = abx * acy - aby * acx

    if cross_product > 0:
        return 1
    else:
        return -1


@jit(nopython=True)
def insideTriangle(px, py, ax, ay, bx, by, cx, cy) -> bool:
    num_turns: int = turnDir(ax, ay, bx, by, px, py) + turnDir(bx, by, cx, cy, px, py) + turnDir(cx, cy, ax, ay, px, py)

    return abs(num_turns) == 3


def asteroid_mas_from_km(diameter_km, distance_AU):
    diameter_mas = 1000 * 206265 * diameter_km / (1.496e8 * distance_AU)
    return diameter_mas


def asteroid_km_from_mas(diameter_mas, distance_AU):
    diameter_km = (diameter_mas / (1000 * 206265)) * 1.496e8 * distance_AU
    return diameter_km


def distance_AU_from_parallax(parallax_arcsec):
    return 8.7882 / parallax_arcsec


def distance_parallax_from_AU(distance_AU):
    return 8.7882 / distance_AU


def star_diameter_km_from_mas(star_mas, asteroid_distance_AU):
    diam_km = 1.496e8 * asteroid_distance_AU * star_mas / (1000 * 206265)
    return diam_km


def star_diameter_mas_from_km(star_km, asteroid_distance_AU):
    diam_mas = star_km * 1000 * 206265 / (1.496e8 * asteroid_distance_AU)
    return diam_mas


def star_diameter_fresnel(wavelength_nm, star_mas, asteroid_distance_AU):
    fresnel_length_km = fresnelLength(wavelength_nm=wavelength_nm, Z_AU=asteroid_distance_AU)
    star_diam_km = star_diameter_km_from_mas(star_mas=star_mas, asteroid_distance_AU=asteroid_distance_AU)
    return star_diam_km / fresnel_length_km


# Used for edge-on-disk models
def fraction_covered(star_radius_km, h_km):
    if h_km < 0 or h_km >= 2 * star_radius_km:
        return 0.0

    t1 = star_radius_km ** 2 * np.arccos((star_radius_km - h_km) / star_radius_km)
    t2 = (star_radius_km - h_km) * np.sqrt(2 * star_radius_km * h_km - h_km ** 2)
    return (t1 - t2) / (np.pi * star_radius_km ** 2)


# This function returns the point of intersection of the extension of D and R limb lines for a given chord.
# The origin (x=0, y=0) of the coordinate system is the center of the chord.
def edgeIntersection(lcp):
    assert lcp.D_limb_angle_degrees > 0
    assert lcp.R_limb_angle_degrees > 0

    # We need to make the triangle point down, so D angles must be set so line slopes down to the right
    D_limb_angle = 180 - lcp.D_limb_angle_degrees
    R_limb_angle = lcp.R_limb_angle_degrees

    # Avoid parallel lines that can never intersect, but maintain symmetry
    if R_limb_angle == D_limb_angle == 90:
        R_limb_angle += .1
        D_limb_angle -= .1

    Rx = lcp.chord_length_km / 2
    Ry = 0

    Dx = -Rx
    Dy = 0

    Cx, Cy = linesIntersectAt(D_limb_angle, Dx, Dy, R_limb_angle, Rx, Ry)

    return Cx, Cy


# Calculate contact points of the star with the edges.
# This allows the easy detection of a graze where the fraction_covered() function is no longer relevant
def contactPoints(LCP):
    # d = 0
    # star_radius = LCP.star_radius_km
    half_chord = LCP.chord_length_km / 2

    # Contact values are relative to the relevant geomettrical edge
    D_first_contact = -LCP.star_radius_km / np.sin(np.radians(LCP.D_limb_angle_degrees))
    D_last_contact = -D_first_contact

    R_first_contact = -LCP.star_radius_km / np.sin(np.radians(LCP.R_limb_angle_degrees))
    R_last_contact = -R_first_contact

    if LCP.debug:
        print(f'D_first_contact: {D_first_contact:0.2f}')
        print(f'D_last_contact: {D_last_contact:0.2f}')
        print(f'R_first_contact: {R_first_contact:0.2f}')
        print(f'R_last_contact: {R_last_contact:0.2f}')

    D_first_km = D_first_contact - half_chord
    D_last_km = D_last_contact - half_chord

    R_first_km = R_first_contact + half_chord
    R_last_km = R_last_contact + half_chord

    return D_first_km, D_last_km, R_first_km, R_last_km


# Used for disk-on-disk model
def areaOfIntersection(Rast, Rstar, d):
    assert d >= 0
    assert d <= Rast + Rstar
    t1 = np.sqrt((-d + Rstar - Rast) * (-d - Rstar + Rast) * (-d + Rstar + Rast) * (d + Rstar + Rast))
    t2 = (d ** 2 + Rstar ** 2 - Rast ** 2) / (2 * d * Rstar)
    t3 = (d ** 2 - Rstar ** 2 + Rast ** 2) / (2 * d * Rast)
    area = Rstar ** 2 * np.arccos(t2) + Rast ** 2 * np.arccos(t3) - t1 / 2
    return area


# Used for disk-on-disk model
def Istar(Rast, Rstar, d):
    assert d >= 0
    starArea = np.pi * Rstar * Rstar
    asteroidArea = np.pi * Rast * Rast
    if Rast >= Rstar and d <= Rast - Rstar:
        # The star is covered by the asteroid
        fractionStarIntensity = 0.0
    elif Rstar >= Rast and d <= Rstar - Rast:
        # The asteroid is inside the star disk
        fractionStarIntensity = (starArea - asteroidArea) / starArea
    elif d <= Rstar + Rast:
        # The disks intersect
        a = areaOfIntersection(Rast, Rstar, d)
        fractionStarIntensity = (starArea - a) / starArea
    else:
        # The star and asteroid do not overlap
        fractionStarIntensity = 1.0
    return fractionStarIntensity


def categorizeEvent(LCP):
    D_first_actual, D_last_actual, R_first_actual, R_last_actual = contactPoints(LCP)

    if LCP.debug:
        print(f'\nD_first_actual: {D_first_actual:0.2f}')
        print(f'D_last_actual: {D_last_actual:0.2f}')
        print(f'R_first_actual: {R_first_actual:0.2f}')
        print(f'R_last_actual: {R_last_actual:0.2f}\n')

    if LCP.miss_distance_km == 0.0:
        if D_last_actual <= R_first_actual:
            return 'normal'
        else:
            return 'graze'
    else:
        if LCP.miss_distance_km < LCP.star_radius_km:
            return 'partial miss'
        else:
            return 'total miss'


def decide_model_to_use(LCP):
    frame_time_km = LCP.frame_time * LCP.shadow_speed

    # print(f'star_diameter_km:  {LCP.star_diameter_km:0.3f}')
    # print(f'frame_time_km:     {frame_time_km:0.3f}\n')

    sum_integrators = LCP.star_diameter_km + frame_time_km

    # print(f'fresnel_length_km: {LCP.fresnel_length_km:0.3f}')
    # print(f'integrators:       {sum_integrators:0.3f}\n')

    if sum_integrators > 2 * LCP.fresnel_length_km and not LCP.star_diameter_km == 0.0:  # This integrates away diffraction wiggles
        if LCP.asteroid_diameter_km >= 10 * LCP.star_diameter_km:
            return 'Use edge-on-disk model because asteroid diameter is at least 10 times the star diameter'
        else:
            return 'Use disk-on-disk model because asteroid diameter is less than 10 time the star diameter'
    elif frame_time_km < 2 * LCP.fresnel_length_km:  # We are sampling fast enough to see diffraction wiggles
        return 'Use diffraction model as conditions allow diffraction effects to be recorded.'
    else:
        return 'Use of square-wave model is indicated.'


def getPlotDandRedgePoints(LCP):
    half_chord = LCP.chord_length_km / 2

    Dtop_y = Rtop_y = LCP.star_diameter_km * 1.5

    assert LCP.D_limb_angle_degrees > 0
    assert LCP.R_limb_angle_degrees > 0

    D_alpha_angle_degrees = 90 - LCP.D_limb_angle_degrees
    R_alpha_angle_degrees = 90 - LCP.R_limb_angle_degrees

    Dtop_x = -Dtop_y * np.tan(np.radians(D_alpha_angle_degrees)) - half_chord

    Rtop_x = Rtop_y * np.tan(np.radians(R_alpha_angle_degrees)) + half_chord

    Dmid_y = Rmid_y = 0
    Dmid_x = -half_chord
    Rmid_x = half_chord

    # If the D and R edges intersect within the range of the plot, set the
    # bottom points at this intersection.

    Cx, Cy = edgeIntersection(LCP)
    # if abs(Cy) < 2 * LCP['star_diameter_km']:
    if True:
        # The intersection is within range
        if LCP.debug:
            print(f'\nThe D and R edges intersect within the plot range')
            print(f'Cx: {Cx:0.2f}  Cy: {Cy:0.2f}\n')
        Dbot_y = Rbot_y = Cy
        Dbot_x = Rbot_x = Cx

    A = Point(Dtop_x, Dtop_y)
    B = Point(Dmid_x, Dmid_y)
    C = Point(Dbot_x, Dbot_y)
    X = Point(Rtop_x, Rtop_y)
    Y = Point(Rmid_x, Rmid_y)
    Z = Point(Rbot_x, Rbot_y)

    return A, B, C, X, Y, Z


# Used by edge-on-disk models
def generatePointsInStar(LCP, pts_in_diameter=52):
    # The default value of 52 for pts_in_diameter yields 2026 points across the star
    r = LCP.star_radius_km
    rsq = r * r
    xvals = np.linspace(-r, r, pts_in_diameter)
    yvals = np.linspace(-r, r, pts_in_diameter)
    # dither = (xvals[1] - xvals[0]) / 3
    x_coord = []
    y_coord = []
    for x in xvals:
        # x += np.random.normal(loc=0.0, scale=dither)      # dither the x value (to suppress jaggies on 45 degree lines)
        for y in yvals:
            # y += np.random.normal(loc=0.0, scale=dither)  # dither the y value
            if x * x + y * y <= rsq:
                x_coord.append(x)
                y_coord.append(y)
    return np.array(x_coord), np.array(y_coord)


# Used by edge-on-disk models
def generateCoveringTriangle(LCP):
    half_chord = LCP.chord_length_km / 2

    Dtop_y = Rtop_y = LCP.star_radius_km

    assert LCP.D_limb_angle_degrees > 0
    assert LCP.R_limb_angle_degrees > 0

    D_alpha_angle_degrees = 90 - LCP.D_limb_angle_degrees
    R_alpha_angle_degrees = 90 - LCP.R_limb_angle_degrees

    Dtop_x = -Dtop_y * np.tan(np.radians(D_alpha_angle_degrees)) - half_chord

    Rtop_x = Rtop_y * np.tan(np.radians(R_alpha_angle_degrees)) + half_chord

    Cx, Cy = edgeIntersection(LCP)

    Ax = Dtop_x
    Ay = Dtop_y
    Bx = Rtop_x
    By = Rtop_y

    return Ax, Ay, Bx, By, Cx, Cy


# Used by edge-on-disk models
@jit(nopython=True)
def fraction_covered_by_triangle(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, star_x, star_y):
    hits: int = 0
    for i in range(star_x.size):
        if insideTriangle(star_x[i], star_y[i], ax, ay, bx, by, cx, cy):
            hits += 1

    return hits / star_x.size


@dataclass
class LightcurveParameters:
    baseline_ADU: float
    bottom_ADU: float

    frame_time: float
    shadow_speed: float

    wavelength_nm: float

    sky_motion_mas_per_sec: float

    sigmaB: float = None  # standard deviation of baseline points (needed for metric)

    magDrop: float = None

    asteroid_diameter_km: float = None
    asteroid_diameter_mas: float = None
    asteroid_rho: float = None

    asteroid_distance_AU: float = None
    asteroid_distance_arcsec: float = None

    fresnel_length_km: float = None   # There is no 'setter' for this parameter
    fresnel_length_sec: float = None  # There is no 'setter' for this.

    star_diameter_mas: float = None
    star_diameter_km: float = None
    star_radius_km: float = None
    star_rho: float = None  # star diameter expressed in fresnel units (no setter)

    D_limb_angle_degrees: float = 89.999  # To avoid possible infinities we avoid 90
    R_limb_angle_degrees: float = 89.999  # "

    chord_length_km: float = None
    chord_length_sec: float = None

    npoints: int = 2048
    miss_distance_km: float = 0.0
    debug: bool = False

    name_list: list[str] = field(default_factory=list)

    def __post_init__(self):
        for name in dir(self):
            exclude_list = ['set', 'check_for_none', 'document', 'asteroid_rho', 'star_rho',
                            'shadow_speed', 'fresnel_length_km', 'name_list', 'wavelength_nm']
            if not (name.startswith('__') or name in exclude_list):
                self.name_list.append(name)

    def set(self, name, value):
        if value is not None and value < 0.0:
            raise ValueError(f'{name} cannot be negative.')

        if name not in self.name_list:
            raise ValueError(f'{name} is not a valid lightcurve parameter name or is not settable after creation.')

        if name == 'asteroid_distance_AU' and value is not None:
            if self.asteroid_distance_arcsec is not None:
                raise ValueError(f'asteroid distance has already been set')
            else:
                if self.sky_motion_mas_per_sec is None:
                    self.sky_motion_mas_per_sec = asteroid_mas_from_km(self.shadow_speed, value)
                else:
                    self.shadow_speed = asteroid_km_from_mas(self.sky_motion_mas_per_sec, value)
                self.fresnel_length_km = fresnelLength(self.wavelength_nm, value)
                self.fresnel_length_sec = self.fresnel_length_km / self.shadow_speed
                self.asteroid_distance_arcsec = distance_parallax_from_AU(value)

        if name == 'asteroid_distance_arcsec' and value is not None:
            if self.asteroid_distance_AU is not None:
                raise ValueError(f'asteroid distance has already been set')
            else:
                self.asteroid_distance_AU = distance_AU_from_parallax(value)
                if self.sky_motion_mas_per_sec is None:
                    self.sky_motion_mas_per_sec = asteroid_mas_from_km(self.shadow_speed, self.asteroid_distance_AU)
                else:
                    self.shadow_speed = asteroid_km_from_mas(self.sky_motion_mas_per_sec, self.asteroid_distance_AU)
                self.fresnel_length_km = fresnelLength(self.wavelength_nm, self.asteroid_distance_AU)
                self.fresnel_length_sec = self.fresnel_length_km / self.shadow_speed

        if name == 'asteroid_diameter_km' and value is not None:
            if self.asteroid_diameter_mas is not None:
                raise ValueError(f'asteroid diameter has already been set')
            elif self.asteroid_distance_AU is None:
                raise ValueError(f'asteroid distance must be set before asteroid diameter.')
            else:
                self.asteroid_diameter_mas = asteroid_mas_from_km(diameter_km=value,
                                                                  distance_AU=self.asteroid_distance_AU)
                rho = (value / 2.0) / self.fresnel_length_km
                self.asteroid_rho = rho

        if name == 'asteroid_diameter_mas' and value is not None:
            if self.asteroid_diameter_km is not None:
                raise ValueError(f'asteroid diameter has already been set')
            elif self.asteroid_distance_AU is None:
                raise ValueError(f'asteroid distance must be set before asteroid diameter.')
            else:
                self.asteroid_diameter_km = asteroid_km_from_mas(diameter_mas=value,
                                                                 distance_AU=self.asteroid_distance_AU)
                rho = self.asteroid_diameter_km / self.fresnel_length_km / 2
                self.asteriod_rho = rho

        if name == 'star_diameter_mas' and value is not None:
            if self.asteroid_distance_AU is None:
                raise ValueError(f'asteroid distance must be specified before star diameter.')
            if self.star_diameter_km is not None:
                raise ValueError(f'star diameter has already been set')
            else:
                self.star_diameter_km = star_diameter_km_from_mas(value, self.asteroid_distance_AU)
                rho = self.star_diameter_km / self.fresnel_length_km
                self.star_rho = rho
                self.star_radius_km = self.star_diameter_km / 2

        if name == 'star_diameter_km' and value is not None:
            if self.asteroid_distance_AU is None:
                raise ValueError(f'asteroid distance must be specified before star diameter.')
            if self.star_diameter_mas is not None:
                raise ValueError(f'star diameter has already been set')
            else:
                self.star_diameter_mas = star_diameter_mas_from_km(value, self.asteroid_distance_AU)
                rho = value / self.fresnel_length_km
                self.star_rho = rho
                self.star_radius_km = value / 2

        if name == 'chord_length_km' and value is not None:
            if self.chord_length_sec is not None:
                raise ValueError(f'chord length has already been set.')
            if self.asteroid_diameter_km is None:
                raise ValueError(f'"shadow_speed" and asteroid size must be set beford chord length')
            else:
                if math.isclose(value, self.asteroid_diameter_km, abs_tol=10**-4):
                    value = self.asteroid_diameter_km
                if value > self.asteroid_diameter_km:
                    raise ValueError(f'chord length specified exceeds asteroid diameter.')
                self.chord_length_sec = value / self.shadow_speed

        if name == 'chord_length_sec' and value is not None:
            if self.chord_length_km is not None:
                raise ValueError(f'chord length has already been set.')
            if self.asteroid_diameter_km is None:
                raise ValueError(f'shadow speed and asteroid size must be set beford chord length')
            else:
                chord_size = value * self.shadow_speed
                if math.isclose(chord_size, self.asteroid_diameter_km, abs_tol=10**-4):
                    chord_size = self.asteroid_diameter_km
                if chord_size <= self.asteroid_diameter_km:
                    self.chord_length_km = value * self.shadow_speed
                else:
                    raise ValueError(f'chord length specified exceeds asteroid diameter.')

        self.__dict__[name] = value

    def check_for_none(self):
        for name in self.name_list:
            if self.__dict__[name] is None:
                return True, name
        return False, 'all needed parameters are set'

    def document(self):
        if self.check_for_none():
            output_str = ['\n!!!! There are values remaining to be set !!!!\n']
        else:
            output_str = []

        output_str.append(f"baseline intensity: {self.baseline_ADU:0.2f} ADU")
        output_str.append(f"bottom   intensity: {self.bottom_ADU:0.2f} ADU\n")

        if self.fresnel_length_km is not None:
            output_str.append(f"fresnel_length: {self.fresnel_length_km:0.2f} km\n")
        else:
            output_str.append(f"fresnel_length: None\n")

        if self.star_diameter_mas is not None:
            output_str.append(f"star_diameter_mas: {self.star_diameter_mas:0.2f} mas")
        else:
            output_str.append(f"star_diameter_mas: None")
        if self.star_diameter_km is not None:
            output_str.append(f"star_diameter_km:  {self.star_diameter_km:0.2f} km")
        else:
            output_str.append(f"star_diameter_km:  None")
        if self.star_radius_km is not None:
            output_str.append(f"star_radius_km:    {self.star_radius_km:0.2f} km")
        else:
            output_str.append(f"star_radius_km:    None")
        if self.star_rho is not None:
            output_str.append(f"star_rho:          {self.star_rho:0.2f} fresnel lengths\n")
        else:
            output_str.append(f"star_rho:          None\n")

        if self.asteroid_diameter_km is not None:
            output_str.append(f"asteroid_diameter_km:  {self.asteroid_diameter_km:0.2f} km")
        else:
            output_str.append(f"asteroid_diameter_km:  None")
        if self.asteroid_diameter_mas is not None:
            output_str.append(f"asteroid_diameter_mas: {self.asteroid_diameter_mas:0.2f} mas")
        else:
            output_str.append(f"asteroid_diameter_mas: None")
        if self.asteroid_rho is not None:
            output_str.append(f"asteroid_rho:          {self.asteroid_rho:0.2f} fresnel lengths\n")
        else:
            output_str.append(f"asteroid_rho:          None\n")

        if self.asteroid_distance_AU is not None:
            output_str.append(f"asteroid_distance_AU:     {self.asteroid_distance_AU:0.2f} AU")
        else:
            output_str.append(f"asteroid_distance_AU:     None")
        if self.asteroid_distance_arcsec is not None:
            output_str.append(f"asteroid_distance_arcsec: {self.asteroid_distance_arcsec:0.2f} arcsec\n")
        else:
            output_str.append(f"asteroid_distance_arcsec: None\n")

        output_str.append(f"shadow_speed: {self.shadow_speed:0.4f} km/sec")
        output_str.append(f"sky_motion:   {self.sky_motion_mas_per_sec:0.4f} mas/sec\n")

        output_str.append(f"frame_time: {self.frame_time:0.4f} sec\n")

        if self.D_limb_angle_degrees is not None:
            output_str.append(f"D_limb_angle_degrees: {self.D_limb_angle_degrees:0.1f} degrees")
        else:
            output_str.append(f"D_limb_angle_degrees: None")

        if self.R_limb_angle_degrees is not None:
            output_str.append(f"R_limb_angle_degrees: {self.R_limb_angle_degrees:0.1f} degrees")
        else:
            output_str.append(f"R_limb_angle_degrees: None")

        if self.chord_length_km is not None:
            output_str.append(f"chord_length_km:  {self.chord_length_km:0.2f} km")
        else:
            output_str.append(f"chord_length_km:  None")
        if self.chord_length_sec is not None:
            output_str.append(f"chord_length_sec: {self.chord_length_sec:0.2f} sec\n")
        else:
            output_str.append(f"chord_length_sec: None\n")

        output_str.append(
            f"miss_distance_km: {self.miss_distance_km:0.2f} (if non-zero, lightcurve is modelled as a partial or complete miss)\n")

        output_str.append(f"wavelength_nm: {self.wavelength_nm} nm")
        output_str.append(f"npoints: {self.npoints} points in model lightcurve")
        # output_str.append(f"debug: {self.debug} (if True, many functions print extra info)")
        output_str.append('\n')

        return output_str


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Vector:
    x: float
    y: float


def coeffStandardLineEquation(alpha_degrees, px, py):
    # standard line equation: a * x + b * y + c = 0
    assert 180 >= alpha_degrees >= 0
    a = np.sin(np.radians(alpha_degrees))
    b = -np.cos(np.radians(alpha_degrees))
    c = -py * b - px * a
    return a, b, c


def linesIntersectAt(angle1, px1, py1, angle2, px2, py2):
    a1, b1, c1 = coeffStandardLineEquation(angle1, px1, py1)
    a2, b2, c2 = coeffStandardLineEquation(angle2, px2, py2)

    denom = a1 * b2 - a2 * b1
    if denom == 0:
        # The lines are parallel and so do not intersect
        return None, None
    else:
        x_intercept = (b1 * c2 - b2 * c1) / denom
        y_intercept = (c1 * a2 - c2 * a1) / denom
        return x_intercept, y_intercept


def y_from_x(x, angle, px, py):
    a, b, c = coeffStandardLineEquation(angle, px, py)
    return -(c + a * x) / b


def plot_diffraction(x, y, first_wavelength, last_wavelength, LCP, figsize=(14, 6),
                     title='Diffraction model plot',
                     showLegend=False, showNotes=False, plot_versus_time=False, zoom=1):
    zoom_factor = zoom
    asteroid_radius_km = LCP.asteroid_diameter_km / 2

    rho = LCP.asteroid_rho

    half_chord = LCP.chord_length_km / 2
    graze_offset_km = np.sqrt((LCP.asteroid_diameter_km / 2) ** 2 - half_chord ** 2)

    if graze_offset_km > asteroid_radius_km:
        left_edge = None
        right_edge = None
    else:
        right_edge = np.sqrt(asteroid_radius_km ** 2 - graze_offset_km ** 2)
        left_edge = -right_edge

    fig = plt.figure(constrained_layout=False, figsize=figsize)
    fig.canvas.manager.set_window_title(title)

    # ax2 = fig.add_subplot(1, 4, (1, 2))  # lightcurve axes
    # ax1 = fig.add_subplot(1, 4, (3, 4))  # event illustration axes
    ax2 = fig.add_subplot(1, 2, 1)  # lightcurve axes
    ax1 = fig.add_subplot(1, 2, 2)  # event illustration axes

    circle1 = patches.Circle((0, 0), radius=asteroid_radius_km, facecolor="None", edgecolor='red')

    right_limit = (3 * asteroid_radius_km / zoom_factor)
    left_limit = - right_limit

    ax1.axis('equal')

    ax1.set_xlim(left_limit, right_limit)
    ax1.set_ylim(left_limit, right_limit)
    ax1.add_patch(circle1)
    ax1.hlines(graze_offset_km, xmin=left_limit * 0.9, xmax=right_limit * 0.9, ls='--', color='black')
    ax1.grid()
    ax1.set_xlabel('Kilometers')
    ax1.set_ylabel('KM')
    ax1.set_title(f'Observation path')

    # Get star disk chords
    star_disk_y = None
    d_chords = None
    if not LCP.star_diameter_km == 0.0:
        d_chords, d_chords_alone, *_ = get_star_chord_samples(x=x, plot_margin=20, LCP=LCP)
        star_disk_y = lightcurve_convolve(sample=d_chords_alone,
                                          lightcurve=y,
                                          shift_needed=len(d_chords_alone) - 1)

        # Block integrate star_disk_y by frame_time to get camera_y
        span_km = x[-1] - x[0]
        resolution_km = span_km / LCP.npoints
        n_sample_points = round(LCP.frame_time * LCP.shadow_speed / resolution_km)
        sample = np.repeat(1.0 / n_sample_points, n_sample_points)
        camera_y = lightcurve_convolve(sample=sample, lightcurve=star_disk_y,
                                       shift_needed=len(sample) - 1)
    else:
        # Block integrate y by frame_time to get camera_y
        span_km = x[-1] - x[0]
        resolution_km = span_km / LCP.npoints
        n_sample_points = round(LCP.frame_time * LCP.shadow_speed / resolution_km)
        sample = np.repeat(1.0 / n_sample_points, n_sample_points)
        camera_y = lightcurve_convolve(sample=sample, lightcurve=y,
                                       shift_needed=len(sample) - 1)

    ax2.set_ylim(-0.1 * LCP.baseline_ADU, 1.5 * LCP.baseline_ADU)

    if plot_versus_time:
        ax2.plot(x / LCP.shadow_speed, y, '-', color='black', label='Underlying')
        ax2.plot(x / LCP.shadow_speed, camera_y, '-', color='red', label='Camera response')
        if not LCP.star_diameter_km == 0.0:
            ax2.plot(x / LCP.shadow_speed, star_disk_y, '-', color='blue', label="Star disk response")
        ax2.set_xlabel('Seconds')
        ax2.set_ylabel('ADU')
    else:
        ax2.plot(x, y, '-', color='black', label='Underlying')
        ax2.plot(x, camera_y, '-', color='red', label='Camera response')
        if not LCP.star_diameter_km == 0.0:
            ax2.plot(x, star_disk_y, '-', color='blue', label="Star disk response")
        ax2.set_xlabel('Kilometers')
        ax2.set_ylabel('ADU')

    # if not LCP.star_diameter_km == 0.0:
    #     ax2.plot(x, star_disk_y, '-', color='blue', label="Star disk response")
        # ax2.plot(x, d_chords * LCP.baseline_ADU, '-', color='orange', label='Star disk chords')

    if left_edge is not None:
        if plot_versus_time:
            ax2.vlines([left_edge / LCP.shadow_speed, right_edge / LCP.shadow_speed], color='red', ymin=0,
                       ymax=1.2 * LCP.baseline_ADU, ls='--', label='Geometric edge')

        else:
            ax2.vlines([left_edge, right_edge], color='red', ymin=0, ymax=1.2 * LCP.baseline_ADU, ls='--',
                       label='Geometric edge')
    else:
        ax2.vlines([-graze_offset_km, graze_offset_km], color='gray', ymin=0, ymax=1.4, ls='--', label='graze position')
        ax2.vlines([-asteroid_radius_km, asteroid_radius_km], color='black',
                   ymin=0, ymax=1.4, ls='--', label='asteroid radius')

    if showLegend:
        ax2.legend(loc='best', fontsize=8)

    rho_adder = ''
    if rho > 32:
        rho_adder = f'(central spot not computed)'

    rho_wavelength = first_wavelength
    if first_wavelength == last_wavelength:
        ax2.set_title(f'Single wavelength diffraction ({first_wavelength} nm) {rho_adder}')
    else:
        rho_wavelength = (first_wavelength + last_wavelength) // 2
        ax2.set_title(f'Integrated diffraction (wavelength range: \n{first_wavelength}nm to '
                      f'{last_wavelength}nm in 10 nm steps)\n{rho_adder}')

    if plot_versus_time:
        ax2.set_xlim(-3 * asteroid_radius_km / zoom_factor / LCP.shadow_speed,
                     3 * asteroid_radius_km / zoom_factor / LCP.shadow_speed)
    else:
        ax2.set_xlim(-3 * asteroid_radius_km / zoom_factor, 3 * asteroid_radius_km / zoom_factor)

    ax2.grid()

    # Add text annotation to plot for asteroid diameter, distance, etc
    s = f'asteroid diameter: {LCP.asteroid_diameter_km:0.2f} km'
    s = s + f'\nasteroid distance: {LCP.asteroid_distance_AU:0.2f} AU'
    s = s + f'\nframe time: {LCP.frame_time:0.3f} sec'
    s = s + f'\ngraze offset: {graze_offset_km:0.2f} km'
    s = s + f'\nrho: {rho:0.2f} @ {rho_wavelength} nm'

    if showNotes:
        ax2.text(0.01, 0.03, s, transform=ax2.transAxes,
                 bbox=dict(facecolor='white', alpha=1), fontsize=10)
    plt.show()


def generalizedDiffraction(LCP, wavelength1=None, wavelength2=None, skip_central_calc=False):
    # A family of central spot lightcurves are calculated when rho <= 32, otherwise the
    # analytic diffraction equation is used to produce the curve family that is
    # subsequently integrated.

    # Single frequency lightcurves can be generated by setting wavelength1 = wavelength2 at
    # the desired wavelength.

    # Grazing observation lightcurves can be generated with any amount of offset, including
    # outside the geometrical shadow.

    asteroid_diam_km = LCP.asteroid_diameter_km
    asteroid_distance_AU = LCP.asteroid_distance_AU

    half_chord = LCP.chord_length_km / 2
    graze_offset_km = np.sqrt((LCP.asteroid_diameter_km / 2) ** 2 - half_chord ** 2)

    y_avg = []
    x_avg = []
    # yz_avg = []
    # xz_avg = []

    rows_y = []
    rows_x = []
    rows_u = []

    wavelength_steps = 10

    N = 2048  # This gives a FOV of 64 fresnel lengths

    max_n_off = N // 2 - 1

    if wavelength1 is None:
        wavelength1 = LCP.wavelength_nm - 100

    if wavelength2 is None:
        wavelength2 = LCP.wavelength_nm + 100

    wavelength1 = int(wavelength1)
    wavelength2 = int(wavelength2)

    rho_wavelength = (wavelength1 + wavelength2) // 2

    if wavelength1 == wavelength2:
        wavelengths = [wavelength1]
        single_wavelength = True
    else:
        wavelengths = [i for i in range(wavelength1, wavelength2 + 1, wavelength_steps)]
        single_wavelength = len(wavelengths) == 1

    asteroid_radius_km = asteroid_diam_km / 2

    # Compose title
    title = f'graze-{graze_offset_km:0.2f}km_astdiam-{asteroid_diam_km:0.2f}km_astdist-{asteroid_distance_AU}AU'
    if not single_wavelength:
        title += f'_{wavelength1}-to-{wavelength2}nm'
    else:
        title += f'_{wavelength1}nm'

    if graze_offset_km == 0:
        right_edge = asteroid_radius_km
        left_edge = -right_edge
    elif graze_offset_km > asteroid_radius_km:
        right_edge = left_edge = None
    else:
        right_edge = np.sqrt(asteroid_radius_km ** 2 - graze_offset_km ** 2)
        left_edge = -right_edge

    asteroid_radius_km = asteroid_diam_km / 2

    rho = asteroid_diam_km / fresnelLength(wavelength_nm=rho_wavelength, Z_AU=asteroid_distance_AU)

    if rho <= 32 and not skip_central_calc:
        for wavelength in wavelengths:
            fresnel_length_km = fresnelLength(wavelength_nm=wavelength, Z_AU=asteroid_distance_AU)
            if graze_offset_km == 0.0:
                n_off = 0
            else:
                n_off = convert_km_distance_to_pixels(graze_offset_km, N, fresnel_length_km)
                if n_off > max_n_off:
                    n_off = max_n_off
            my_field, fresnel_length_km, L_km, x_fl = basicCalcField(N, fresnel_length_km, diam_km=asteroid_diam_km)
            transform = (fft2(my_field)) / N
            image = abs(transform) ** 2
            row = image[N // 2 + n_off, :]
            x_km = x_fl * fresnel_length_km

            rows_y.append(row)
            rows_x.append(x_km)
            rows_u.append(x_fl)
    else:
        plot_span = 3
        x_km = np.linspace(-plot_span * asteroid_radius_km, plot_span * asteroid_radius_km, 1000)
        z_km = np.sqrt(graze_offset_km ** 2 + x_km ** 2)

        for wavelength in wavelengths:
            fresnel_length_km = fresnelLength(wavelength_nm=wavelength, Z_AU=asteroid_distance_AU)

            x_fl = x_km / fresnel_length_km

            z_fl = z_km / fresnel_length_km

            y = []
            asteroid_radius_u = asteroid_radius_km / fresnel_length_km
            for u_value in z_fl:
                y.append(analyticDiffraction(u_value - asteroid_radius_u, 'R'))

            rows_y.append(y)
            rows_x.append(x_km)
            rows_u.append(x_fl)

    k = len(rows_x)

    # Generate an interpolation function f for each wavelength sample (not needed for analyticDiffraction,
    # but it doesn't slow the calculation in a perceptible way)
    interpolation_function = []
    for i in range(k):
        interpolation_function.append(interpolate.interp1d(rows_x[i], rows_y[i], kind='quadratic', bounds_error=True))

    # Perform integration (using interpolation functions generated above)
    for x in rows_x[0]:
        x_avg.append(x)
        my_sum = 0.0
        for i in range(k):
            # interpolate so that we can sum contributions at identical positions
            my_sum += interpolation_function[i](x)
        y_avg.append(my_sum / k)

    y_ADU = scaleToADU(np.array(y_avg), LCP=LCP)
    # y_ADU = np.array(y_avg)

    return np.array(x_avg), y_ADU, left_edge, right_edge


@jit(nopython=True)
def tresterModifiedAperture(aperture, N):
    modified_aperture = zeros((N, N), dtype=complex128)
    c1 = 1.0j * pi / N
    for i in prange(N):
        for k in prange(N):
            c2 = i * i + k * k
            modified_aperture[i, k] = aperture[i, k] * exp(c1 * c2)
    return modified_aperture


def fresnelLength(wavelength_nm, Z_AU):
    wavelength = wavelength_nm * 1e-9 * 1e-3
    Z = Z_AU * 150e6
    return np.sqrt(wavelength * Z / 2)


def fresnelDiffraction(aperture, N):
    modified_aperture = tresterModifiedAperture(aperture, N)
    transform = (fft2(modified_aperture)) / N
    return abs(transform) ** 2


# The following conversion functions are useful for annotating image plots

def convert_fresnel_distance_to_pixels(fresnel_distance, N):
    return round(fresnel_distance * np.sqrt(N / 2))


def convert_km_distance_to_pixels(km_distance, N, fresnel_length_km):
    return convert_fresnel_distance_to_pixels(km_distance / fresnel_length_km, N)


# We use Numba to greatly speedup the calculation of the modified aperture.
@jit(nopython=True)
def basicCalcField(N, fresnel_length_km, diam_km):
    # None of the following values are used in the computation of the modified aperture field
    # in the plane of the asteroid, but are needed to interpret the result in useful
    # physical parameters of field-of-view (in kilometers) and fresnel lengths, so we compute
    # these values and provide them in the returned values

    elz = 2 * fresnel_length_km * fresnel_length_km
    L_km = np.sqrt(elz * N)  # km  (We're viewing an L_km x L_km image) == fresnel_length_km * sqrt(2*N) == FOV
    # For N = 2048 the FOV = 64 fresnel lengths on a side
    x = np.linspace(-L_km / 2, L_km / 2, N)
    x_fl = x / fresnel_length_km  # x_fl is x expressed in fresnel lengths

    my_field = np.zeros((N, N), dtype=np.complex128)  # Ensure the field is complex

    amplitude = 1.0 + 0.0j

    r = diam_km / 2.0
    r2 = r * r

    c1 = 1j * pi / N

    for i in range(N):
        for j in range(N):
            c2 = i * i + j * j
            in_target = (x[j] * x[j] + x[i] * x[i]) < r2
            in_target = in_target or (x[j] * x[j] + x[i] * x[i]) < r2
            if not in_target:
                my_field[i, j] = amplitude * np.exp(c1 * c2)

    return my_field, fresnel_length_km, L_km, x_fl


# The following routine is used in this paper to generate 'analytic knife edge' diffraction curves
def diffraction_u(u: float, type_D_or_R: str) -> float:
    """
    Calculate normalized knife edge diffraction curve for a single wavelength using fresnel unit scale.

    When type_D_or_R == 'R' then a 'reappearance' curve
    results: it is 'dark' to the left of zero, and 'light' to the right of zero.

    When type_D_or_R == 'D' then a 'disappearance' curve
    results: it is 'light' to the left of zero, and 'dark' to the right of zero.

    :param u: fresnel unit
    :param type_D_or_R: either 'D' (disappearance) or 'R' (reappearance)
    :return: Normalized, single wavelength, diffraction intensity as a function of fresnel units
    """
    assert type_D_or_R in ['D', 'R'], 'Invalid type code given to diffraction_u'

    if type_D_or_R == 'D':
        u = -u
    ssa, csa = fresnel(u)
    return 0.5 * ((0.5 + csa) ** 2 + (0.5 + ssa) ** 2)


def analyticDiffraction(u_value, D_or_R):
    return diffraction_u(u_value, D_or_R)


def demo_diffraction_field(LCP, title_adder='',
                           figsize=(8, 5)):
    if LCP.asteroid_rho > 32:
        raise Exception(
            f'The asteroid rho is {LCP.asteroid_rho:0.2f} which is greater than 32, the max that we can display.')

    title = f'Diffraction pattern on the ground: {title_adder}'
    N = 2048

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    fig.canvas.manager.set_window_title(title)
    ax1 = fig.add_subplot()

    my_field, fresnel_length_km, L_km, x_fl = basicCalcField(N=N, fresnel_length_km=LCP.fresnel_length_km,
                                                             diam_km=LCP.asteroid_diameter_km)
    transform = (fft2(my_field)) / N
    image = abs(transform) ** 2
    # row = image[N // 2, :]
    # x_km = x_fl * fresnel_length_km

    ax1.matshow(image[512:1536, 512:1536], cmap='gray')

    circle1 = patches.Circle((512, 511),
                             radius=convert_km_distance_to_pixels(LCP.asteroid_diameter_km / 2, N, fresnel_length_km),
                             facecolor="None", edgecolor='red', linewidth=2)
    ax1.add_patch(circle1)
    plt.show()


def dodModel(margin, LCP):
    star_radius = LCP.star_radius_km
    asteroid_radius = LCP.asteroid_diameter_km / 2

    assert LCP.asteroid_diameter_km >= LCP.chord_length_km

    half_chord = LCP.chord_length_km / 2
    if LCP.miss_distance_km == 0:
        y_offset = np.sqrt(asteroid_radius ** 2 - half_chord ** 2)
    else:
        y_offset = asteroid_radius + LCP.miss_distance_km

    half_width = star_radius + asteroid_radius + margin

    x = np.linspace(-half_width, half_width, 2048)
    dvalues = np.sqrt(x ** 2 + y_offset ** 2)

    # max_d = np.max(dvalues)

    EntryContact_x = ExitContact_x = None
    for i, d in enumerate(dvalues):
        if EntryContact_x is None:
            if d <= star_radius + asteroid_radius:
                EntryContact_x = x[i]
        elif ExitContact_x is None and x[i] > 0:
            if d >= asteroid_radius + star_radius:
                ExitContact_x = x[i]

    Itemp = []
    for d in dvalues:
        Itemp.append(Istar(asteroid_radius, star_radius, d))

    y_ADU = scaleToADU(np.array(Itemp), LCP=LCP)
    # y_ADU = np.array(Itemp)

    return x, y_ADU, dvalues, EntryContact_x, ExitContact_x


def dodLightcurve(LCP):
    x, y_ADU, _, _, _ = dodModel(margin=10, LCP=LCP)

    half_chord = LCP.chord_length_km / 2
    R_edge = half_chord
    D_edge = - R_edge

    return x, y_ADU, D_edge, R_edge


def illustrateDiskOnDiskEvent(LCP: LightcurveParameters, axes,
                              showLegend=False, showNotes=False):
    half_chord = LCP.chord_length_km / 2
    asteroid_radius = LCP.asteroid_diameter_km / 2

    if LCP.miss_distance_km > 0:
        half_chord = 0

    y_offset = np.sqrt(asteroid_radius ** 2 - half_chord ** 2)
    starY = -LCP.miss_distance_km
    starX = -half_chord - 0.6 * LCP.star_diameter_km

    asteroidY = y_offset
    asteroidX = 0.0

    if not LCP.miss_distance_km == 0:
        if LCP.miss_distance_km <= LCP.star_radius_km:
            event_type_str = 'partial miss'
        else:
            event_type_str = 'miss'
    elif y_offset <= asteroid_radius - LCP.star_radius_km:
        event_type_str = 'normal'
    elif y_offset <= LCP.star_radius_km - asteroid_radius:
        event_type_str = 'annular'
    else:
        event_type_str = 'graze'

    # Create the title for the plot
    title_msg = 'The star is moving from left to right into the asteroid.\n'
    axes.set_title(title_msg)

    # Create and place the star image
    star_image = patches.Circle((starX, starY), radius=LCP.star_radius_km, facecolor="yellow", edgecolor='red')
    axes.axis('equal')
    axes.add_patch(star_image)

    # Put gray dot in center of star
    axes.plot(starX, starY, marker='o', color='gray')

    # Plot the star path
    axes.hlines(starY, xmin=starX, xmax=-0.0 * LCP.star_diameter_km, ls='-.',
                color='blue', label='star path')
    axes.hlines(starY, xmax=2.5 * LCP.star_diameter_km, xmin=0.0 * LCP.star_diameter_km, ls='-.', color='blue')

    # Create and place the asteroid image
    asteroid_image = patches.Circle((asteroidX, asteroidY), radius=LCP.asteroid_diameter_km / 2, facecolor="lightgray",
                                    edgecolor='gray')
    axes.add_patch(asteroid_image)

    # Put gray dot in center of asteroid
    axes.plot(asteroidX, asteroidY, marker='o', color='gray')

    # Plot chord
    axes.plot([-half_chord, half_chord], [0, 0], color='black', linewidth=4, label='chord')

    axes.set_xlabel("Kilometers")
    axes.grid()

    scale = 2 * LCP.asteroid_diameter_km
    axes.set_ylim(-scale, scale)
    axes.set_xlim(-scale, scale)
    axes.set_ylabel('KM')

    # Add legend to identify the lines
    if showLegend:
        axes.legend(loc='best', fontsize=10)

    # Add text annotation to plot for asteroid diameter, distance, etc
    s = f"star diameter: {LCP.star_diameter_km:0.2f} km"
    s = s + f"\nasteroid diameter: {LCP.asteroid_diameter_km:0.2f} km"
    s = s + f"\nasteroid distance: {LCP.asteroid_distance_AU:0.2f} AU"
    s = s + f"\nframe time: {LCP.frame_time:0.3f} sec"
    if LCP.miss_distance_km == 0:
        s = s + f"\nchord length: {LCP.chord_length_km:0.2f} km"
    s = s + f"\n\nThis is a {event_type_str} event."
    if event_type_str == 'partial miss' or event_type_str == 'total miss':
        s = s + f"\n  miss distance: {LCP.miss_distance_km:0.2f} km"
    s = s + f"\n\nChord end points are defined\nas where star center hits edge"
    margin = 0.02
    if showNotes:
        axes.text(1.0 - margin, margin, s,
                  horizontalalignment='right',
                  verticalalignment='bottom',
                  transform=axes.transAxes, bbox=dict(facecolor='white', alpha=1), fontsize=8)


def plot_disk_on_disk(x, y, LCP, figsize=(10, 6),
                      title='Disk on disk model plot',
                      showLegend=False, showNotes=False, plot_versus_time=False):
    # Block integrate y by frame_time
    span_km = x[-1] - x[0]
    resolution_km = span_km / LCP.npoints
    n_sample_points = round(LCP.frame_time * LCP.shadow_speed / resolution_km)
    sample = np.repeat(1.0 / n_sample_points, n_sample_points)
    camera_y = lightcurve_convolve(sample=sample, lightcurve=y, shift_needed=len(sample) - 1)

    # scaleToADU(camera_y, LCP=LCP)
    # scaleToADU(y, LCP=LCP)

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    fig.canvas.manager.set_window_title(title)

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    if plot_versus_time:
        ax1.plot(x / LCP.shadow_speed, y, label='Underlying', ls='-')
        ax1.plot(x / LCP.shadow_speed, camera_y, label="Camera response")
        ax1.set_xlabel('Seconds')
        ax1.set_ylabel('ADU')
    else:
        ax1.plot(x, y, label='Underlying', ls='-')
        ax1.plot(x, camera_y, label="Camera response")
        ax1.set_xlabel('Kilometers')
        ax1.set_ylabel('ADU')

    ax1.grid()
    if plot_versus_time:
        half_chord = (LCP.chord_length_km / 2) / LCP.shadow_speed
    else:
        half_chord = LCP.chord_length_km / 2
    if LCP.miss_distance_km == 0:
        ax1.vlines([-half_chord, half_chord], 0, 1.1 * LCP.baseline_ADU, color='red', ls='--', label='Geometric edges')

    if showLegend:
        ax1.legend(loc='best', fontsize=10)
    ax1.set_ylim(0, 1.2 * LCP.baseline_ADU)

    illustrateDiskOnDiskEvent(LCP, showLegend=showLegend, showNotes=showNotes, axes=ax2)
    plt.show()


def illustrateEdgeOnDiskEvent(LCP: LightcurveParameters, axes,
                              showLegend=False, showNotes=False, demo_covering_triangle=False):
    A, B, C, X, Y, Z = getPlotDandRedgePoints(LCP)

    Cx, Cy = edgeIntersection(LCP)  # The point where the D and R edges intersect

    event_type_str = categorizeEvent(LCP)

    D_first_actual, D_last_actual, R_first_actual, R_last_actual = contactPoints(LCP)

    starY = 0.0
    starX = D_first_actual - 0.1 * LCP.star_diameter_km

    if LCP.chord_length_km > 2 * LCP.star_diameter_km:
        add_gray_divider = True
        # Move the origin of the D line left by one star diameter
        A.x -= LCP.star_diameter_km
        B.x -= LCP.star_diameter_km
        C.x -= LCP.star_diameter_km
        # Move the star too
        starX -= LCP.star_diameter_km

        # Move the origin of the R line right by one star diameter
        X.x += LCP.star_diameter_km
        Y.x += LCP.star_diameter_km
        Z.x += LCP.star_diameter_km
    else:
        add_gray_divider = False

    # Plot the D and R edges
    axes.plot([A.x, B.x, C.x], [A.y, B.y, C.y], marker='', color='green', linewidth=4, label='D edge')
    axes.plot([X.x, Y.x, Z.x], [X.y, Y.y, Z.y], marker='', color='red', linewidth=4, label='R edge')

    # For a 'miss', draw a gray line to show virtual edge of asteroid
    if not LCP.miss_distance_km == 0:
        axes.plot([C.x, Z.x], [C.y, C.y], marker='', color='gray', linewidth=4)

    # Create the title for the plot
    title_msg = 'The star is moving from left to right.\n'
    if add_gray_divider and LCP.miss_distance_km == 0:
        title_msg += 'The chord versus star size is too large to plot.'
        title_msg += '\nThis is indicated by the gray separator.'
    axes.set_title(title_msg)

    if event_type_str == 'partial miss' or event_type_str == 'total miss':
        starY = -(abs(Cy) + LCP.miss_distance_km)

    # Create and place the star image
    star_image = patches.Circle((starX, starY), radius=LCP.star_radius_km, facecolor="yellow", edgecolor='red')
    axes.axis('equal')
    axes.add_patch(star_image)

    # Put gray dot in center of star
    axes.plot(starX, starY, marker='o', color='gray')

    # Plot the star path
    axes.hlines(starY, xmin=starX, xmax=-0.0 * LCP.star_diameter_km, ls='-.',
                color='blue', label='star path')
    axes.hlines(starY, xmax=2.5 * LCP.star_diameter_km, xmin=0.0 * LCP.star_diameter_km, ls='-.', color='blue')

    # Plot chord (when there is one)
    if event_type_str == 'normal' or event_type_str == 'graze':
        axes.plot([B.x, Y.x], [0, 0], color='black', linewidth=4, label='chord')
        if add_gray_divider:
            axes.set_xticks([])
            axes.vlines(0, ymin=-2 * LCP.star_diameter_km, ymax=2 * LCP.star_diameter_km,
                        color='lightgray', linewidth=30)

    # Set the plot x and y dimensions (scaled by star diameter or chord, whichever is larger)
    if LCP.star_diameter_km > LCP.chord_length_km:
        scale = LCP.star_diameter_km
    else:
        scale = LCP.chord_length_km

    axes.set_ylim(-3 * scale, 3 * scale)
    axes.set_xlim(-3 * scale, 3 * scale)
    axes.set_xlabel("Kilometers")
    axes.set_ylabel('KM')

    if event_type_str == 'partial miss' or event_type_str == 'total miss':
        # Translate plot so that star is in bottom third. If chord will be off screen,
        # we need to use the star to scale the plot
        if abs(starY) > 3 * LCP.star_diameter_km:
            scale = LCP.star_diameter_km
        axes.set_ylim(-3 * scale + starY, 3 * scale - starY)

    # Add legend to identify the lines
    if showLegend:
        axes.legend(loc='best', fontsize=10)

    # Add text annotation to plot for asteroid diameter, distance, etc
    s = f"star diameter: {LCP.star_diameter_km:0.2f} km"
    s = s + f"\nasteroid diameter: {LCP.asteroid_diameter_km:0.2f} km"
    s = s + f"\nasteroid distance: {LCP.asteroid_distance_AU:0.2f} AU"
    if event_type_str == 'partial miss' or event_type_str == 'total miss':
        s = s + f"\nchord length: None"
    else:
        s = s + f"\nchord length: {LCP.chord_length_km:0.2f} km"
    s = s + f"\nD limb angle: {LCP.D_limb_angle_degrees} degrees"
    s = s + f"\nR limb angle: {LCP.R_limb_angle_degrees} degrees"
    s = s + f"\n\nThis is a {event_type_str} event."
    if event_type_str == 'partial miss' or event_type_str == 'total miss':
        s = s + f"\n  miss distance: {LCP.miss_distance_km:0.2f} km"
    s = s + f"\n\nChord end points are defined\nas where star center hits edge"
    margin = 0.02
    if showNotes:
        axes.text(1.0 - margin, margin, s,
                  horizontalalignment='right',
                  verticalalignment='bottom',
                  transform=axes.transAxes,
                  bbox=dict(facecolor='white', alpha=1), fontsize=8)

    # Demo the star starting position and its covering triangle (that which occults the star)
    if demo_covering_triangle and not add_gray_divider:
        Ax, Ay, Bx, By, Cx, Cy = generateCoveringTriangle(LCP)  # Returns the three points that define the triangle
        axes.plot([Ax, Bx], [Ay, By], color='lightblue', linewidth=6)
        axes.plot([Bx, Cx], [By, Cy], color='lightblue', linewidth=6)
        axes.plot([Cx, Ax], [Cy, Ay], color='lightblue', linewidth=6)


def eodModel(LCP: LightcurveParameters, star_master_x, star_master_y):
    # Even in a graze D_first_actual and R_last_actual are still usable.
    D_first_actual, D_last_actual, R_first_actual, R_last_actual = contactPoints(LCP)
    dvalues = np.linspace(D_first_actual - 10, R_last_actual + 10, LCP.npoints)

    if LCP.miss_distance_km > 0:
        Cx, Cy = edgeIntersection(LCP)
        translated_star_master_y = star_master_y - LCP.miss_distance_km + Cy
    else:
        translated_star_master_y = star_master_y

    Ax, Ay, Bx, By, Cx, Cy = generateCoveringTriangle(LCP)

    # half_chord = LCP.chord_length_km / 2

    y = []
    x = []
    for d in dvalues:
        x.append(d)
        if d < D_first_actual:
            y.append(1.0)
        elif D_first_actual <= d <= R_last_actual:
            translated_star_master_x = star_master_x + d
            y.append(1.0 - fraction_covered_by_triangle(Ax, Ay, Bx, By, Cx, Cy, translated_star_master_x,
                                                        translated_star_master_y))
        else:
            y.append(1.0)

    y_array = np.array(y)
    y_ADU = scaleToADU(y_array, LCP=LCP)
    # y_ADU = np.array(y)

    return np.array(x), y_ADU


def eodLightcurve(LCP):
    star_master_x, star_master_y = generatePointsInStar(LCP)
    x, y_ADU = eodModel(LCP, star_master_x, star_master_y)

    if LCP.miss_distance_km == 0.0:
        half_chord = LCP.chord_length_km / 2
        D_edge = -half_chord
        R_edge = half_chord
    else:
        D_edge = R_edge = None

    return x, y_ADU, D_edge, R_edge


def plot_edge_on_disk(x, y, D_edge, R_edge, LCP, figsize=(10, 6),
                      title='Edge on disk model plot',
                      showLegend=False, showNotes=False, plot_versus_time=False):
    # Block integrate y by frame_time
    span_km = x[-1] - x[0]
    resolution_km = span_km / LCP.npoints
    n_sample_points = round(LCP.frame_time * LCP.shadow_speed / resolution_km)
    sample = np.repeat(1.0 / n_sample_points, n_sample_points)
    camera_y = lightcurve_convolve(sample=sample, lightcurve=y, shift_needed=len(sample) - 1)

    # scaleToADU(camera_y, LCP=LCP)
    # scaleToADU(y, LCP=LCP)

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    fig.canvas.manager.set_window_title(title)

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    if plot_versus_time:
        ax1.plot(x / LCP.shadow_speed, y, label='Underlying')
        ax1.plot(x / LCP.shadow_speed, camera_y, label='Camera response')
        ax1.set_xlabel('Seconds')
    else:
        ax1.plot(x, y, label='Underlying')
        ax1.plot(x, camera_y, label='Camera response')
        ax1.set_xlabel('Kilometers')

    # half_chord = LCP.chord_length_km / 2
    if LCP.miss_distance_km == 0:
        if plot_versus_time:
            ax1.vlines([D_edge / LCP.shadow_speed, R_edge / LCP.shadow_speed], ymin=0, ymax=1.1 * LCP.baseline_ADU,
                       ls='--', color='blue', label='Asteroid edge')
        else:
            ax1.vlines([D_edge, R_edge], ymin=0, ymax=1.1 * LCP.baseline_ADU, ls='--', color='blue',
                       label='Asteroid edge')
    ax1.set_ylim(0, 1.2 * LCP.baseline_ADU)
    if showLegend:
        ax1.legend(loc='best', fontsize=10)

    ax1.grid()

    illustrateEdgeOnDiskEvent(LCP, showLegend=showLegend, showNotes=showNotes,
                              axes=ax2, demo_covering_triangle=False)
    plt.show()


def demo_event(LCP, model, title='Generic model', showLegend=False, showNotes=False,
               plot_versus_time=False, plots_wanted=True):
    if model == 'disk-on-disk':
        x, y, D_edge, R_edge = dodLightcurve(LCP=LCP)
        if plots_wanted:
            plot_disk_on_disk(x=x, y=y, LCP=LCP, title=title,
                              showLegend=showLegend, showNotes=showNotes,
                              plot_versus_time=plot_versus_time)
    elif model == 'edge-on-disk':
        x, y, D_edge, R_edge = eodLightcurve(LCP=LCP)
        if plots_wanted:
            plot_edge_on_disk(x=x, y=y, D_edge=D_edge, R_edge=R_edge, LCP=LCP,
                              title=title, showLegend=showLegend, showNotes=showNotes,
                              plot_versus_time=plot_versus_time)
    elif model == 'diffraction':
        x, y, D_edge, R_edge = generalizedDiffraction(LCP=LCP, wavelength1=None, wavelength2=None,
                                                      skip_central_calc=False)
        wavelength1 = LCP.wavelength_nm - 100
        wavelength2 = LCP.wavelength_nm + 100
        if plots_wanted:
            plot_diffraction(x=x, y=y, first_wavelength=wavelength1,
                             last_wavelength=wavelength2, LCP=LCP, title=title,
                             showLegend=showLegend, showNotes=showNotes, plot_versus_time=plot_versus_time)
    else:
        raise Exception(f"Model '{model}' is unknown.")

    return x, y, D_edge, R_edge


def timeSampleLightcurve(x_km, y_ADU, D_km, R_km, LCP, start_time=0):
    # Convert x to times
    x_sec = x_km / LCP.shadow_speed

    # convert D_km and R_km to time
    D_sec = D_km / LCP.shadow_speed
    R_sec = R_km / LCP.shadow_speed

    # Set time of points so that first point is at time 0.0
    offset = -x_sec[0]
    x_sec += offset
    D_sec += offset
    R_sec += offset

    # Build interpolation function
    interpolator = interpolate.interp1d(x_sec, y_ADU)

    # Compute the time (seconds) desired for first point
    sample_time = -x_sec[0] + start_time

    x_vals = []
    y_vals = []

    while sample_time <= x_sec[-1]:
        x_vals.append(sample_time)
        y_vals.append(interpolator(sample_time))
        sample_time += LCP.frame_time

    return np.array(x_vals), np.array(y_vals), D_sec, R_sec


def addShotNoise(y, level=1):
    for i in range(y.size):
        y[i] += level * np.random.normal(loc=0.0, scale=np.sqrt(y[i]))


def addReadNoise(y, sigma):
    for i in range(y.size):
        y[i] += np.random.normal(loc=0.0, scale=sigma)


def formatTime(t_sec):
    hours = t_sec // (60 * 60)
    t_sec -= hours * 60 * 60
    minutes = t_sec // 60
    t_sec -= minutes * 60
    return f'[{hours:02.0f}:{minutes:02.0f}:{t_sec:0.4f}]'


def getLightcurveParametersInCsvForm(LCP):
    lcp_lines = []
    for line in LCP.document():
        if line.endswith('\n'):
            lcp_lines.append(f'# {line[:-1]}\n')
            lcp_lines.append(f'#\n')
        else:
            lcp_lines.append(f'# {line}\n')
    return lcp_lines


def versionLightcurveModeller():
    return "Lightcurve modeller version: 1.0"


# def writeCsv(x_pts, y_pts, D_sec, R_sec, LCP, shot_noise, read_noise, model, folder, filename):
#     csv_lines = []
#
#     csv_lines.append(f'# from file: {filename}\n')
#     csv_lines.append(f'#\n')
#
#     csv_lines.append(f'# D @ {formatTime(D_sec)}\n')
#     csv_lines.append(f'# R @ {formatTime(R_sec)}\n')
#     csv_lines.append(f'#\n')
#
#     csv_lines.append(f'# lightcurve model: {model}\n')
#     csv_lines.append(f'# event category: {categorizeEvent(LCP)}\n')
#
#     utc_time = datetime.now(timezone.utc)
#
#     csv_lines.append(f'#\n')
#     csv_lines.append(f'# UTC time: {utc_time}   {versionLightcurveModeller()}\n')
#     csv_lines.append(f'#\n')
#
#     # Document lightcurve parameters
#     lcp_lines = getLightcurveParametersInCsvForm(LCP=LCP)
#     csv_lines += lcp_lines
#
#     csv_lines.append(f'FrameNum,timeInfo,signal-Target\n')
#
#     frame_num = 0
#     for i in range(len(x_pts)):
#         csv_lines.append(f'{frame_num},{formatTime(x_pts[i])},{y_pts[i]:0.2f}\n')
#
#     data_folder = Path(folder)
#     filepath = data_folder / filename
#
#     with open(filepath, 'w') as f:
#         f.writelines(csv_lines)
#
#     return csv_lines
