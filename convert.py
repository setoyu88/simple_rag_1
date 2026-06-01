"""PDF → Markdown 変換ツール。

doclingを用いてPDFファイルをMarkdown形式に変換するCLIスクリプト。
オプションで画像抽出とOpenAI APIによる画像説明文生成も行う。
"""

import argparse
import base64
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


@dataclass
class ConversionConfig:
    """変換処理の全設定を保持するデータクラス。

    Args:
        input_pdf: 変換対象PDFファイルのパス。
        output_dir: 出力先ディレクトリのパス。
        extract_images: 画像抽出を有効化するか。
        describe_images: 画像説明文生成を有効化するか。
        model_name: 使用するOpenAIモデル名。
        image_dir: 画像保存先サブディレクトリ名。
        verbose: 詳細ログ出力を有効化するか。
    """

    input_pdf: Path
    output_dir: Path
    extract_images: bool = False
    describe_images: bool = False
    model_name: str = "gpt-5-mini"
    image_dir: str = "images"
    verbose: bool = False


@dataclass
class ExtractedImage:
    """抽出された1枚の画像とそのメタデータを保持する。

    Args:
        index: ドキュメント内の連番（1始まり）。
        file_path: 保存先ファイルパス。
        relative_path: Markdown参照用の相対パス。
        description: LLM生成の説明文。未生成時はNone。
        description_failed: API呼び出し失敗フラグ。
    """

    index: int
    file_path: Path
    relative_path: str
    description: str | None = None
    description_failed: bool = False


@dataclass
class ConversionResult:
    """変換処理全体の結果を保持する。

    Args:
        markdown_path: 生成されたMarkdownファイルのパス。
        images: 抽出された画像のリスト。
        failed_descriptions: 説明文生成に失敗した画像の数。
    """

    markdown_path: Path
    images: list[ExtractedImage] = field(default_factory=list)
    failed_descriptions: int = 0


def _log_info(message: str, verbose: bool) -> None:
    """詳細ログをSTDERRに出力する。

    Args:
        message: ログメッセージ。
        verbose: Trueの場合のみ出力する。
    """
    if verbose:
        print(f"[INFO] {message}", file=sys.stderr)


def _log_warning(message: str) -> None:
    """警告メッセージをSTDERRに出力する。

    Args:
        message: 警告メッセージ。
    """
    print(f"[WARNING] {message}", file=sys.stderr)


def _log_error(message: str) -> None:
    """エラーメッセージをSTDERRに出力する。

    Args:
        message: エラーメッセージ。
    """
    print(f"[ERROR] {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析して返す。

    Returns:
        解析済みの引数オブジェクト。
    """
    parser = argparse.ArgumentParser(
        description="PDFファイルをMarkdown形式に変換するツール"
    )
    parser.add_argument("input_pdf", type=Path, help="変換対象PDFファイルのパス")
    parser.add_argument(
        "output_dir", type=Path, help="出力先ディレクトリのパス（存在しない場合は自動作成）"
    )
    parser.add_argument(
        "--extract-images",
        action="store_true",
        default=False,
        help="図・数式をPNG画像として保存し、Markdownから相対パス参照する",
    )
    parser.add_argument(
        "--describe-images",
        action="store_true",
        default=False,
        help="LLMによる画像説明文をMarkdownに追記する（--extract-images必須）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        dest="model_name",
        help="使用するOpenAIモデル名（デフォルト: gpt-5-mini）",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="images",
        dest="image_dir",
        help="画像保存先サブディレクトリ名（デフォルト: images）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="詳細ログをSTDERRに出力する",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> ConversionConfig:
    """引数を検証して ConversionConfig を生成する。

    バリデーションルール:
    - input_pdf は存在するファイルであること
    - describe_images=True の場合、extract_images=True であること
    - describe_images=True の場合、OPENAI_API_KEY が設定されていること

    Args:
        args: parse_args() が返した引数オブジェクト。

    Returns:
        検証済みの ConversionConfig。

    Raises:
        SystemExit: 入力ファイルが存在しない場合（exit code 1）。
        SystemExit: OPENAI_API_KEY が未設定の場合（exit code 3）。
    """
    input_pdf: Path = args.input_pdf
    if not input_pdf.exists():
        _log_error(f"ファイルが見つかりません: {input_pdf}")
        sys.exit(1)
    if not input_pdf.is_file():
        _log_error(f"指定されたパスはファイルではありません: {input_pdf}")
        sys.exit(1)

    describe_images: bool = args.describe_images
    extract_images: bool = args.extract_images

    if describe_images and not extract_images:
        _log_warning(
            "--describe-images は --extract-images なしでは使用できません。"
            "--describe-images を無効化します。"
        )
        describe_images = False

    if describe_images and not os.environ.get("OPENAI_API_KEY"):
        _log_error("OPENAI_API_KEY が設定されていません。")
        sys.exit(3)

    return ConversionConfig(
        input_pdf=input_pdf,
        output_dir=args.output_dir,
        extract_images=extract_images,
        describe_images=describe_images,
        model_name=args.model_name,
        image_dir=args.image_dir,
        verbose=args.verbose,
    )


def convert_pdf(config: ConversionConfig) -> tuple[str, object]:
    """PDFをMarkdownテキストに変換する。

    Args:
        config: 変換設定。

    Returns:
        (markdown_text, docling_doc) のタプル。

    Raises:
        SystemExit: 変換に失敗した場合（exit code 2）。
    """
    _log_info(f"PDF変換開始: {config.input_pdf}", config.verbose)

    if config.extract_images:
        pipeline_options = PdfPipelineOptions(
            generate_picture_images=True,
            images_scale=2,
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    else:
        converter = DocumentConverter()

    try:
        result = converter.convert(str(config.input_pdf))
    except Exception as e:
        _log_error(f"PDF変換に失敗しました: {e}")
        sys.exit(2)

    markdown_text = result.document.export_to_markdown()
    _log_info("PDF変換完了", config.verbose)
    return markdown_text, result


def extract_images(
    config: ConversionConfig, docling_result: object
) -> list[ExtractedImage]:
    """PDFから画像を抽出して保存する。

    Args:
        config: 変換設定。
        docling_result: convert_pdf() が返した docling の変換結果。

    Returns:
        抽出された画像の ExtractedImage リスト。
    """
    images_dir = config.output_dir / config.input_pdf.stem / config.image_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    _log_info(f"画像抽出開始: 保存先={images_dir}", config.verbose)

    extracted: list[ExtractedImage] = []
    index = 1

    for element, _level in docling_result.document.iterate_items():
        from docling.datamodel.document import PictureItem

        if not isinstance(element, PictureItem):
            continue

        filename = f"picture-{index}.png"
        file_path = images_dir / filename
        relative_path = f"./{config.image_dir}/{filename}"

        try:
            image = element.get_image(docling_result.document)
            if image is not None:
                image.save(str(file_path))
                _log_info(f"画像保存: {file_path}", config.verbose)
                extracted.append(
                    ExtractedImage(
                        index=index,
                        file_path=file_path,
                        relative_path=relative_path,
                    )
                )
                index += 1
        except Exception as e:
            _log_warning(f"画像 {filename} の保存に失敗しました: {e}")

    _log_info(f"画像抽出完了: {len(extracted)} 枚", config.verbose)
    return extracted


def describe_image(
    config: ConversionConfig, image: ExtractedImage
) -> str | None:
    """OpenAI APIを用いて画像の説明文を生成する。

    Args:
        config: 変換設定（model_name を使用）。
        image: 説明文を生成する対象の ExtractedImage。

    Returns:
        生成された説明文。API呼び出し失敗時はNone。
    """
    import openai

    _log_info(
        f"画像説明文生成開始: {image.file_path.name} (モデル: {config.model_name})",
        config.verbose,
    )

    try:
        image_data = base64.b64encode(image.file_path.read_bytes()).decode("utf-8")
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "この画像の内容を日本語で簡潔に説明してください。",
                        },
                    ],
                }
            ],
        )
        description = response.choices[0].message.content
        _log_info(f"画像説明文生成完了: {image.file_path.name}", config.verbose)
        return description
    except openai.APIError as e:
        _log_warning(f"画像 {image.file_path.name} の説明文生成に失敗しました: {e}")
        return None


def build_markdown(
    markdown_text: str,
    images: list[ExtractedImage],
) -> str:
    """Markdownテキストに画像参照と説明文を統合する。

    画像参照を `![picture-N](./images/picture-N.png)` 形式で埋め込み、
    description が設定されている場合は画像参照直下に追記する。

    Args:
        markdown_text: convert_pdf() が返した基本Markdownテキスト。
        images: extract_images() が返した ExtractedImage リスト。

    Returns:
        画像参照と説明文が統合されたMarkdownテキスト。
    """
    if not images:
        return markdown_text

    lines = markdown_text.splitlines()
    result_lines: list[str] = []
    image_index = 0

    for line in lines:
        result_lines.append(line)
        if image_index < len(images) and "<!-- image -->" in line.lower():
            img = images[image_index]
            result_lines.append(f"![picture-{img.index}]({img.relative_path})")
            if img.description:
                result_lines.append(img.description)
            image_index += 1

    remaining_images = images[image_index:]
    for img in remaining_images:
        result_lines.append(f"![picture-{img.index}]({img.relative_path})")
        if img.description:
            result_lines.append(img.description)

    return "\n".join(result_lines)


def save_markdown(
    config: ConversionConfig, markdown_text: str
) -> Path:
    """MarkdownテキストをファイルとしてOUTPUT_DIRに保存する。

    出力ディレクトリを自動作成し、同名ファイルが存在する場合は警告を出して上書きする。
    ファイル名は入力PDFのstemに .md を付けたもの。

    Args:
        config: 変換設定。
        markdown_text: 保存するMarkdownテキスト。

    Returns:
        保存されたMarkdownファイルの絶対パス。
    """
    pdf_subdir = config.output_dir / config.input_pdf.stem
    pdf_subdir.mkdir(parents=True, exist_ok=True)
    output_path = pdf_subdir / f"{config.input_pdf.stem}.md"

    if output_path.exists():
        _log_warning(f"出力先に同名ファイルが存在します。上書きします: {output_path.resolve()}")

    output_path.write_text(markdown_text, encoding="utf-8")
    _log_info(f"Markdown保存完了: {output_path.resolve()}", config.verbose)
    return output_path.resolve()


def main() -> None:
    """CLIエントリーポイント。

    引数を解析・検証し、PDFをMarkdownに変換して保存する。
    成功時はMarkdownファイルの絶対パスをSTDOUTに出力する。
    """
    args = parse_args()
    config = validate_args(args)

    _log_info(
        f"変換開始: input={config.input_pdf} output_dir={config.output_dir}",
        config.verbose,
    )

    markdown_text, docling_result = convert_pdf(config)

    images: list[ExtractedImage] = []
    if config.extract_images:
        images = extract_images(config, docling_result)

        if config.describe_images:
            failed_count = 0
            for img in images:
                description = describe_image(config, img)
                if description is not None:
                    img.description = description
                else:
                    img.description_failed = True
                    failed_count += 1

    final_markdown = build_markdown(markdown_text, images)
    markdown_path = save_markdown(config, final_markdown)

    print(str(markdown_path))


if __name__ == "__main__":
    main()
