from numpy import zeros, complex128
from numba import jit, prange
import numpy as np
from scipy.fftpack import fft2
from numpy import exp, pi
from scipy.special import fresnel
from scipy import interpolate


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
    Z = Z_AU * 149.6e6
    return np.sqrt(wavelength * Z / 2)


def convert_fresnel_distance_to_pixels(fresnel_distance, N):
    return round(fresnel_distance * np.sqrt(N / 2))


def convert_km_distance_to_pixels(km_distance, N, fresnel_length_km):
    return convert_fresnel_distance_to_pixels(km_distance / fresnel_length_km, N)


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
    return 0.5 * ((0.5 + csa)**2 + (0.5 + ssa)**2)


def analyticDiffraction(u_value, side='R'):
    return diffraction_u(u_value, side)


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

    field = np.zeros((N, N), dtype=np.complex128)  # Ensure the field is complex

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
                field[i, j] = amplitude * np.exp(c1 * c2)

    return field, fresnel_length_km, L_km, x_fl


def generalizedDiffraction(asteroid_diam_km, asteroid_distance_AU,
                           graze_offset_km=0, wavelength1=400, wavelength2=600,
                           skip_central_flash=False):
    # A family of central spot lightcurves are calculated when rho <= 32, otherwise the
    # analytic diffraction equation is used to produce the curve family that is
    # subsequently integrated.

    # Single frequency lightcurves can be generated by setting wavelength1 = wavelength2 at
    # the desired wavelength.

    # Grazing observation lightcurves can be generated with any amout of offset, including
    # outside the geometrical shadow.

    y_avg = []
    x_avg = []

    rows_y = []
    rows_x = []
    rows_u = []

    wavelength_steps = 10

    N = 2048  # This gives a FOV of 64 fresnel lengths

    max_n_off = N // 2 - 1

    wavelength1 = int(wavelength1)
    wavelength2 = int(wavelength2)

    rho_wavelength = (wavelength1 + wavelength2) // 2

    if wavelength1 == wavelength2:
        wavelengths = [wavelength1]
        single_wavelength = True
    else:
        wavelengths = [i for i in range(wavelength1, wavelength2 + 1, wavelength_steps)]
        single_wavelength = len(wavelengths) == 1

    # asteroid_radius_km = asteroid_diam_km / 2

    # Compose title
    title = f'graze-{graze_offset_km:0.2f}km_astdiam-{asteroid_diam_km:0.2f}km_astdist-{asteroid_distance_AU}AU'
    if not single_wavelength:
        title += f'_{wavelength1}-to-{wavelength2}nm'
    else:
        title += f'_{wavelength1}nm'

    asteroid_radius_km = asteroid_diam_km / 2

    rho = asteroid_diam_km / fresnelLength(wavelength_nm=rho_wavelength, Z_AU=asteroid_distance_AU)

    if rho <= 32 and not skip_central_flash:
        for wavelength in wavelengths:
            fresnel_length_km = fresnelLength(wavelength_nm=wavelength, Z_AU=asteroid_distance_AU)
            if graze_offset_km == 0.0:
                n_off = 0
            else:
                n_off = convert_km_distance_to_pixels(graze_offset_km, N, fresnel_length_km)
                if n_off > max_n_off:
                    n_off = max_n_off
            field, fresnel_length_km, L_km, x_fl = basicCalcField(N, fresnel_length_km, diam_km=asteroid_diam_km)
            transform = (fft2(field)) / N
            image = abs(transform) ** 2
            row = image[N // 2 + n_off, :]
            x_km = x_fl * fresnel_length_km

            rows_y.append(row)
            rows_x.append(x_km)
            rows_u.append(x_fl)
    else:
        plot_span = 3
        x_km = np.linspace(-plot_span * asteroid_radius_km, plot_span * asteroid_radius_km, N)
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
        cum_sum = 0.0
        for i in range(k):
            # interpolate so that we can sum contributions at identical positions
            cum_sum += interpolation_function[i](x)
        y_avg.append(cum_sum / k)

    return np.array(x_avg), np.array(y_avg), rho, rho_wavelength, wavelengths[0], wavelengths[-1], title


# ############################## Utility functions ###############################
def asteroid_mas(diameter_km, distance_AU):
    milli_arcseconds = 1000 * 206265 * diameter_km / (1.496e8 * distance_AU)
    return milli_arcseconds


def distance_AU_from_parallax_arcsec(parallax_arcsec):
    return 8.7882 / parallax_arcsec


def projected_star_diameter_km(star_mas, asteroid_distance_AU):
    diam = 1.496e8 * asteroid_distance_AU * star_mas / (1000 * 206265)
    return diam


def star_diameter_fresnel(wavelength_nm, star_mas, asteroid_distance_AU):
    fresnel_length_km = fresnelLength(wavelength_nm=wavelength_nm, Z_AU=asteroid_distance_AU)
    star_diam_km = projected_star_diameter_km(star_mas=star_mas, asteroid_distance_AU=asteroid_distance_AU)
    return star_diam_km / fresnel_length_km


# Here we codify the decision process as to which lightcurve model to use
def decide_model_to_use(asteroid_diameter_km=5,
                        asteroid_distance_AU=2,
                        star_diameter_mas=0.0,
                        shadow_speed=5,
                        frame_time=0.0334):

    print(f'asteroid_diameter_km: {asteroid_diameter_km:0.3f}')
    star_diameter_km = projected_star_diameter_km(star_diameter_mas, asteroid_distance_AU)
    print(f'star_diameter_km: {star_diameter_km:0.3f}')

    fresnel_length = fresnelLength(wavelength_nm=500, Z_AU=asteroid_distance_AU)
    print(f'fresnel_length: {fresnel_length:0.3f}')

    fresnel_time_sec = fresnel_length / shadow_speed
    print(f'\n')
    print(f'fresnel_time_sec: {fresnel_time_sec:0.3f}')
    print(f'frame_time: {frame_time:0.3f} sec')
    print(f'\n')

    if star_diameter_km > fresnel_length:  # This integrates away diffraction wiggles
        if asteroid_diameter_km >= 10 * star_diameter_km:
            print('Use edge-on-disk model because asteroid diameter is at least 10 times the star diameter')
        else:
            print('Use disk-on-disk model because asteroid diameter is less than 10 time the star diameter')
    elif frame_time < fresnel_time_sec:  # We are sampling fast enough to see diffraction wiggles
        print('Use diffraction model')
    else:
        print('Use square-wave model because diffraction effects are integrated away')