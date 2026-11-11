#!/usr/bin/env python3
"""Place street photos using coarse colour cues and broad regional priors.

The entry decodes JPEG DC coefficients into an approximate block-colour grid,
extracts climate and vegetation features, and ranks aggregated regional
profiles. It combines the top two matches with one world-centroid vote, then
uses a deterministic spherical geometric median for the final coordinate.
"""

import json
import math
import os
import struct
import urllib.request


CENTROID = (22.17, 10.51)
FEATURES = ['bluesky','skyBlue','skySat','midB','B','midGreen','midG','gray','gndB','yellow','white','gndG','gndR','gndRed','lrgrad','Ysd']
MEAN_STD = [(0.314,.255),(53.074,46.266),(54.184,45.14),(128.298,43.44),(129.422,27.526),(16.863,20.781),(124.34,33.521),(.156,.136),(93.788,44.216),(.004,.015),(.074,.1),(99.676,47.782),(94.938,48.582),(-4.738,14.744),(1.372,32.331),(58.526,12.834)]

# Each row is latitude, longitude, and the aggregate feature profile.
ANCHORS = [
(-37.68,-67.81,[.31,60.51,60.51,149.63,155.65,38.48,143.59,.19,141.57,.01,.09,170,175.13,5.14,-7.96,40.65]),
(-44.16,169.86,[.33,64.94,64.94,76.29,102.62,11.35,76.13,.11,74.25,0,.06,76.8,69.52,-7.28,-23.52,70.25]),
(-25.05,-67.41,[.27,84.85,84.85,117.98,140.4,-.16,126.67,.11,133.65,0,0,144.59,141.77,-2.82,8.97,41.97]),
(-19.29,-45.41,[.15,34.16,34.29,110.46,118.43,11.95,110.35,.17,71.07,.01,.14,79.76,77.43,-2.33,-4.19,63.96]),
(-13.48,40.34,[.47,137.83,137.83,178.56,165.35,66.22,159.1,.05,120.83,.02,.04,142.84,133.4,-9.44,-9.26,47.59]),
(-15.59,147.89,[.35,72.03,72.03,97.83,112.21,14.97,111.91,.04,69.52,.02,.01,88.03,84.72,-3.31,-9.58,57.96]),
(22.37,-96.98,[.31,27.74,29.85,158.43,129.52,11.83,151.95,.2,68.47,0,.08,68.59,66.98,-1.62,-30.04,63.64]),
(7.9,-6.6,[.35,61.31,61.31,125.08,118.09,26.61,112.55,.05,60.99,0,.03,66.21,58.57,-7.64,16.27,63.47]),
(8.22,34.62,[.39,47.12,50.96,106.74,133.51,.16,112.7,.04,139.84,.03,.09,141.6,145.28,3.68,-11.76,56.71]),
(15.13,87.64,[.18,24.37,29.6,107.37,115.09,10.35,110.12,.22,100.58,0,.06,105.53,102.51,-3.02,4.26,58.82]),
(50.17,-124.14,[.34,41.03,42.54,123.82,99.52,5.01,112.88,.11,48.24,0,.02,42.72,41.09,-1.63,35.11,53.83]),
(42.57,-94.31,[.4,62.94,64.1,144.71,149.52,24.75,138.1,.15,136.77,0,.01,144.57,138.86,-5.71,13.21,47.14]),
(46.19,33.4,[.27,45.51,45.85,139.12,137.54,12.85,137.62,.21,99.86,0,.12,100.79,92.91,-7.88,8.22,64.96]),
(46.8,71.19,[.18,22.2,22.62,106.49,114.95,7.55,107.25,.27,73.26,0,.12,81.05,79.05,-2,3.91,63.35]),
(40.47,133.83,[.27,44.47,44.9,120.77,116.07,10,118.19,.23,78.01,0,.08,76.29,71.86,-4.42,-19.46,64.88]),
(62.23,-136.51,[.74,144.32,144.32,213.75,178.61,50.56,168.89,.03,95.85,0,0,106.6,111.39,4.8,9.98,53.31]),
(67.01,22.46,[.5,63.45,64.56,122.55,122.61,17.42,110.24,.09,77.71,0,.05,79.25,64.71,-14.55,14.59,54.48]),
]


class BitReader:
    """Read JPEG entropy-coded bits, skipping byte-stuffing after 0xFF."""

    def __init__(self, data, position):
        self.data = data
        self.position = position
        self.buffer = 0
        self.available = 0

    def read(self, count):
        while self.available < count:
            if self.position >= len(self.data):
                byte = 255
            else:
                byte = self.data[self.position]
                self.position += 1
                if byte == 255:
                    if self.position < len(self.data) and self.data[self.position] == 0:
                        self.position += 1
                    else:
                        byte = 255
            self.buffer = (self.buffer << 8) | byte
            self.available += 8
        self.available -= count
        value = (self.buffer >> self.available) & ((1 << count) - 1)
        self.buffer &= (1 << self.available) - 1 if self.available else 0
        return value


def build_huffman_table(counts, symbols):
    table = {}
    code = 0
    index = 0
    for length, count in enumerate(counts, 1):
        for _ in range(count):
            table[(length, code)] = symbols[index]
            index += 1
            code += 1
        code <<= 1
    return table


def decode_huffman_symbol(reader, table):
    code = 0
    for length in range(1, 17):
        code = (code << 1) | reader.read(1)
        if (length, code) in table:
            return table[(length, code)]
    raise ValueError('invalid Huffman code')


def receive_extended(reader, count):
    if not count:
        return 0
    value = reader.read(count)
    return value - (1 << count) + 1 if value < (1 << (count - 1)) else value


def parse_jpeg_headers(data):
    """Return JPEG quantization, Huffman, frame, and scan metadata."""
    if data[:2] != b'\xff\xd8':
        raise ValueError('not JPEG')
    quantization = {}
    dc_tables = {}
    ac_tables = {}
    components = []
    position = 2

    while position < len(data) - 1:
        if data[position] != 255:
            position += 1
            continue
        marker = data[position + 1]
        if marker in (0xd8, 0xd9) or 0xd0 <= marker <= 0xd7:
            position += 2
            continue
        length = int.from_bytes(data[position + 2:position + 4], 'big')
        segment = data[position + 4:position + 2 + length]

        if marker == 0xdb:
            offset = 0
            while offset < len(segment):
                spec = segment[offset]
                offset += 1
                table_id = spec & 15
                if spec >> 4:
                    quantization[table_id] = [
                        int.from_bytes(segment[offset + 2 * k:offset + 2 * k + 2], 'big')
                        for k in range(64)
                    ]
                    offset += 128
                else:
                    quantization[table_id] = list(segment[offset:offset + 64])
                    offset += 64
        elif marker == 0xc0:
            height, width = struct.unpack('>HH', segment[1:5])
            component_count = segment[5]
            for k in range(component_count):
                sampling = segment[7 + 3 * k]
                components.append((
                    segment[6 + 3 * k],
                    sampling >> 4,
                    sampling & 15,
                    segment[8 + 3 * k],
                ))
        elif marker == 0xc4:
            offset = 0
            while offset < len(segment):
                spec = segment[offset]
                offset += 1
                counts = list(segment[offset:offset + 16])
                offset += 16
                symbols = list(segment[offset:offset + sum(counts)])
                offset += sum(counts)
                target = dc_tables if spec >> 4 == 0 else ac_tables
                target[spec & 15] = build_huffman_table(counts, symbols)
        elif marker == 0xda:
            scan_count = segment[0]
            scan = []
            for k in range(scan_count):
                component_id = segment[1 + 2 * k]
                selectors = segment[2 + 2 * k]
                scan.append((component_id, selectors >> 4, selectors & 15))
            return (width, height, components, quantization, dc_tables,
                    ac_tables, scan, position + 2 + length)
        position += 2 + length
    raise ValueError('missing scan')


def decode_dc_blocks(data):
    """Decode baseline JPEG DC values, skipping AC coefficients safely."""
    (width, height, components, quantization, dc_tables, ac_tables,
     scan, scan_start) = parse_jpeg_headers(data)
    max_h = max(component[1] for component in components)
    max_v = max(component[2] for component in components)
    mcu_cols = (width + 8 * max_h - 1) // (8 * max_h)
    mcu_rows = (height + 8 * max_v - 1) // (8 * max_v)
    grids = [
        [[0] * (mcu_cols * component[1]) for _ in range(mcu_rows * component[2])]
        for component in components
    ]
    component_indices = {component[0]: i for i, component in enumerate(components)}
    reader = BitReader(data, scan_start)
    previous_dc = [0] * len(components)

    for mcu_y in range(mcu_rows):
        for mcu_x in range(mcu_cols):
            for component_id, dc_selector, ac_selector in scan:
                component_index = component_indices[component_id]
                component = components[component_index]
                for block_y in range(component[2]):
                    for block_x in range(component[1]):
                        category = decode_huffman_symbol(reader, dc_tables[dc_selector])
                        previous_dc[component_index] += receive_extended(reader, category)
                        grids[component_index][mcu_y * component[2] + block_y][
                            mcu_x * component[1] + block_x
                        ] = previous_dc[component_index] * quantization[component[3]][0]

                        coefficient = 0
                        while coefficient < 63:
                            symbol = decode_huffman_symbol(reader, ac_tables[ac_selector])
                            if symbol == 0:
                                break
                            run, size = symbol >> 4, symbol & 15
                            if size:
                                reader.read(size)
                            coefficient += 16 if symbol == 240 else run + 1

    return width, height, components, grids


def dc_blocks_to_rgb(width, height, components, grids):
    """Upsample component DC grids and convert their means to approximate RGB."""
    max_h = max(component[1] for component in components)
    max_v = max(component[2] for component in components)
    mcu_cols = (width + 8 * max_h - 1) // (8 * max_h)
    mcu_rows = (height + 8 * max_v - 1) // (8 * max_v)
    rows = mcu_rows * max_v
    columns = mcu_cols * max_h
    output = []
    for row_index in range(rows):
        row = []
        for column_index in range(columns):
            values = []
            for component_index in range(min(3, len(components))):
                grid = grids[component_index]
                grid_row = min(row_index * len(grid) // rows, len(grid) - 1)
                grid_column = min(column_index * len(grid[0]) // columns, len(grid[0]) - 1)
                values.append(grid[grid_row][grid_column] / 8 + 128)
            y_value = values[0]
            cb = values[1] - 128 if len(values) > 1 else 0
            cr = values[2] - 128 if len(values) > 2 else 0
            row.append((
                max(0, min(255, y_value + 1.402 * cr)),
                max(0, min(255, y_value - .34414 * cb - .71414 * cr)),
                max(0, min(255, y_value + 1.772 * cb)),
            ))
        output.append(row)
    return output


def decode_jpeg_rgb(data):
    """Decode JPEG DC block means into a low-resolution RGB grid."""
    width, height, components, grids = decode_dc_blocks(data)
    return dc_blocks_to_rgb(width, height, components, grids)


def mean_rgb(cells):
    return tuple(sum(pixel[channel] for pixel in cells) / len(cells) for channel in range(3))


def band_statistics(rgb_grid):
    """Return RGB means for the sky, middle, and ground image bands."""
    row_count = len(rgb_grid)

    def band(start, end):
        first = int(row_count * start)
        last = max(int(row_count * end), first + 1)
        cells = [pixel for row in rgb_grid[first:last] for pixel in row]
        return mean_rgb(cells)

    return band(0, .18), band(.30, .65), band(.72, 1)


def colour_fractions(cells):
    """Count blue, green, white, grey, and yellow coarse colour categories."""
    blue = green = white = gray = yellow = 0
    for red, green_channel, blue_channel in cells:
        blue += blue_channel > red + 20 and blue_channel > green_channel + 8
        green += green_channel > red + 8 and green_channel > blue_channel + 5
        white += (
            red > 195 and green_channel > 195 and blue_channel > 195
            and max(red, green_channel, blue_channel) - min(red, green_channel, blue_channel) < 25
        )
        gray += (
            max(red, green_channel, blue_channel) - min(red, green_channel, blue_channel) < 14
            and 50 < (red + green_channel + blue_channel) / 3 < 175
        )
        yellow += (
            red > 140 and green_channel > 105 and blue_channel < 110
            and red - blue_channel > 50
        )
    count = len(cells)
    return blue / count, green / count, white / count, gray / count, yellow / count


def luminance_features(rgb_grid, cells):
    """Return whole-image luminance spread and left-to-right gradient."""
    row_count = len(rgb_grid)
    column_count = len(rgb_grid[0])
    luminance = [.299 * r + .587 * g + .114 * b for r, g, b in cells]
    average = sum(luminance) / len(luminance)
    spread = math.sqrt(sum((value - average) ** 2 for value in luminance) / len(luminance))

    left = [
        .299 * r + .587 * g + .114 * b
        for row in rgb_grid for r, g, b in row[:column_count // 2]
    ]
    right = [
        .299 * r + .587 * g + .114 * b
        for row in rgb_grid for r, g, b in row[column_count // 2:]
    ]
    return sum(left) / len(left) - sum(right) / len(right), spread


def extract_features(rgb_grid):
    """Extract the fixed ordered feature vector used by the regional priors."""
    cells = [pixel for row in rgb_grid for pixel in row]
    sky, middle, ground = band_statistics(rgb_grid)
    blue, green_fraction, white, gray, yellow = colour_fractions(cells)
    left_right_gradient, luminance_spread = luminance_features(rgb_grid, cells)
    return [
        blue,
        sky[2] - sky[0],
        max(sky) - min(sky),
        middle[2],
        sum(pixel[2] for pixel in cells) / len(cells),
        middle[1] - middle[0],
        middle[1],
        gray,
        ground[2],
        yellow,
        white,
        ground[1],
        ground[0],
        ground[0] - ground[1],
        left_right_gradient,
        luminance_spread,
    ]


def score_anchor(feature_vector, profile):
    """Return standardized squared distance between photo and anchor profiles."""
    return sum(
        (
            (feature_vector[index] - MEAN_STD[index][0]) / MEAN_STD[index][1]
            - (profile[index] - MEAN_STD[index][0]) / MEAN_STD[index][1]
        ) ** 2
        for index in range(len(feature_vector))
    )


def rank_anchors(feature_vector):
    """Sort regional centres by their colour-profile distance to the photo."""
    scored = [
        (score_anchor(feature_vector, profile), (latitude, longitude))
        for latitude, longitude, profile in ANCHORS
    ]
    return sorted(scored)


def spherical_distance(point_a, point_b):
    latitude_a, longitude_a = point_a
    latitude_b, longitude_b = point_b
    phi_a = math.radians(latitude_a)
    phi_b = math.radians(latitude_b)
    delta_latitude = math.radians(latitude_b - latitude_a)
    delta_longitude = math.radians(longitude_b - longitude_a)
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_longitude / 2) ** 2
    )
    return 12742 * math.asin(min(1, math.sqrt(haversine)))


def spherical_geometric_median(points):
    """Compute a deterministic Weiszfeld median over spherical distances."""
    latitude = sum(point[0] for point in points) / len(points)
    longitude = sum(point[1] for point in points) / len(points)
    for _ in range(50):
        denominator = weighted_latitude = weighted_longitude = 0.0
        for point in points:
            distance = spherical_distance((latitude, longitude), point)
            if distance < 1e-9:
                return point
            weight = 1.0 / distance
            denominator += weight
            weighted_latitude += point[0] * weight
            weighted_longitude += point[1] * weight
        new_latitude = weighted_latitude / denominator
        new_longitude = weighted_longitude / denominator
        if abs(new_latitude - latitude) < 1e-6 and abs(new_longitude - longitude) < 1e-6:
            break
        latitude, longitude = new_latitude, new_longitude
    return latitude, longitude


def predict_location(jpeg_bytes):
    """Decode a photo, rank priors, and combine top-2 plus one centroid vote."""
    features = extract_features(decode_jpeg_rgb(jpeg_bytes))
    matches = rank_anchors(features)
    candidates = [coordinate for _, coordinate in matches[:2]] + [CENTROID]
    return spherical_geometric_median(candidates)


def http_request(url, payload=None):
    body = None if payload is None else json.dumps(payload, separators=(',', ':')).encode()
    headers = {'Content-Type': 'application/json'} if body else {}
    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    environment_url = os.environ.get('MAKEFUN_ENV_URL')
    if not environment_url:
        raise SystemExit('MAKEFUN_ENV_URL is required')
    base_url = environment_url.rstrip('/')
    photo_list = json.loads(http_request(base_url + '/photos'))
    if not photo_list.get('photos'):
        raise SystemExit('The environment returned no photos')
    answers = []
    for photo in photo_list['photos']:
        try:
            latitude, longitude = predict_location(http_request(base_url + photo['url']))
        except Exception:
            latitude, longitude = CENTROID
        answers.append({
            'id': photo['id'],
            'lat': round(latitude, 6),
            'lon': round(longitude, 6),
        })
    http_request(base_url + '/answers', {'answers': answers})
    for answer in answers:
        print(json.dumps({'lat': answer['lat'], 'lon': answer['lon']}))


if __name__ == '__main__':
    main()
