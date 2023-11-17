import re
import argparse

def extract_buffers_info(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    pattern = r'=====TorchInductor Stream Scheduler Tree stream allocation=====(.*?)=====TorchInductor Stream Scheduler Tree stream allocation end====='
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return {}

    relevant_content = matches[0]

    buffer_pattern = r'torch\._inductor\.stream_scheduler:\[INFO\] (\w+) (\d+) (\d+)'
    buffer_matches = re.findall(buffer_pattern, relevant_content)

    buffer_info = {}
    for match in buffer_matches:
        buffer_name, stream_id, volume = match
        buffer_info[buffer_name] = (stream_id, volume)

    return buffer_info

def compare_buffers_info(file1, file2, output_path=None):
    info1 = extract_buffers_info(file1)
    info2 = extract_buffers_info(file2)

    differences = []
    for buffer_name, (stream1, volume1) in info1.items():
        stream2, volume2 = info2.get(buffer_name, (None, None))
        if stream1 != stream2:
            differences.append(f"Buffer: {buffer_name}, Stream IDs: origin: {stream1}, bypass: {stream2}, Volume: {volume1}")

    if output_path:
        with open(output_path, 'w') as f:
            for diff in differences:
                f.write(diff + '\n')
    else:
        for diff in differences:
            print(diff)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare buffer information between two files.")
    parser.add_argument('--origin', help="Path to the original stream assignment")
    parser.add_argument('--bypass', help="Path to the bypass stream assignment")
    parser.add_argument('-o', '--output', help="Path to the output file", required=False)

    args = parser.parse_args()
    
    compare_buffers_info(args.origin, args.bypass, args.output)
