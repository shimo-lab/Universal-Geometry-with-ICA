def convert_to_japanese_with_tags(input_path, output_path):
    """
    Convert an English README with Japanese translations in comments, delimited by
    specific start and end tags, to a Japanese README where Japanese text is primary
    and the English text is commented out.

    Parameters:
    input_path (str): Path to the input README file.
    output_path (str): Path to the output README.ja.md file.
    """

    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_path, 'w', encoding='utf-8') as file:
        in_translation_block = False

        for line in lines:
            if '<!-- TRANSLATION_START -->' in line:
                in_translation_block = True
                nested_translation_block = True
                en_lines = []
                ja_lines = []
                continue  # Skip the start tag line
            elif '<!-- TRANSLATION_END -->' in line:
                # check if both English and Japanese translations are present
                en_lines = [l_.strip() for l_ in en_lines if l_.strip()]
                ja_lines = [l_.strip() for l_ in ja_lines if l_.strip()]
                if not en_lines or not ja_lines:
                    print('\n'.join(en_lines))
                    print('\n'.join(ja_lines))
                    raise ValueError(
                        'Both English and Japanese translations must be present.')
                in_translation_block = False
                continue  # Skip the end tag line

            if in_translation_block:
                if line.strip().startswith('<!--'):
                    nested_translation_block = False
                    # This is the start of the Japanese translation
                    file.write(line.replace('<!--', ''))
                elif line.strip().endswith('--->'):
                    # This is the end of the Japanese translation, write it without the comment markers
                    file.write(line.replace('--->', ''))
                else:
                    if nested_translation_block:
                        en_lines.append(line)
                        file.write(f'<!--\n{line}--->\n')
                    else:
                        ja_lines.append(line)
                        file.write(line)

            else:
                # Outside of translation blocks, just copy the line
                file.write(line)


# Example usage:
convert_to_japanese_with_tags('README.md', 'README.ja.md')
