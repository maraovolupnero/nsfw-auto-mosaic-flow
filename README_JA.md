# NSFW Auto Mosaic Flow

[English](README.md) | **日本語**

Ultralytics YOLO SegmentationモデルでNSFW画像内の対象領域を検出し、フォルダ単位でPixelateモザイクを一括適用するWindows向けデスクトップアプリです。NVIDIA GPUがなくてもCPUで利用できます。

作者: [maraovolupnero](https://github.com/maraovolupnero) / ライセンス: [MIT](LICENSE)

> [!WARNING]
> NSFW画像を扱う用途を想定しています。自動検出は完全ではないため、公開前に必ず出力画像を目視確認してください。

## 主な機能

- `.pt` YOLOモデルと対象クラスIDの指定
- PNG / JPG / JPEG / WebPのフォルダ一括処理
- Segmentation Maskを優先し、Mask膨張・Feather後にPixelateモザイクを合成
- MaskがないDetectionモデルではBBox楕円マスクへ自動フォールバック
- `model.names`からクラス一覧を読み込み、ID・名前・チェックボックスで対象を選択
- Confidence、Pixel Size、Expand、Featherの調整（推論解像度は1280固定）
- 遠い・小さい対象向けの重なり付き分割による高精度検出
- モデル向け推奨設定へのワンクリック復帰と、各スライダーでの微調整
- Original / Detection / Processedプレビュー
- 検出なし画像の未加工コピーと同名ファイルの連番保存
- 出力先が空の場合、入力フォルダ内へ`mosaic_output`（既存時は連番）を自動作成
- 停止要求、進捗、処理キュー、`logs/result.csv`への全検出クラス・マスク対象クラスの分離記録
- 処理キューの画像を選択してOriginal / Detection / Processedを個別確認
- Processed画像へブラシで追加モザイク、消しゴム、元に戻す、保存
- プレビュー上のマウスホイールで処理キューの前後画像へ移動
- プレビュー右側の縦スライダーで25%から400%まで拡大縮小
- 拡大中は左ドラッグで画像を移動（Processedのブラシ利用中は右ドラッグ）
- 処理済み画像を1画像1ページのPDFへまとめて出力
- GPU自動利用とCUDAエラー時のCPU再試行

## 推奨画面サイズ

このツールは`1280×800`以上のウィンドウサイズを基準に設計しています。初期表示は`1420×940`です。

ウィンドウを小さくして使用することもできますが、設定欄やプレビューが狭くなり、操作性や視認性が低下します。通常はウィンドウを最大化するか、初期サイズに近い大きさで使用してください。

Detection表示ではSegmentation Maskの輪郭を表示します。対象クラスは膨張後の輪郭を緑色、対象外クラスは元Mask輪郭を灰色で表示します。Maskがない旧DetectionモデルだけBBoxと楕円範囲を表示します。

`Expand 15%`はSegmentation MaskへMorphology Dilateを適用して約15%膨張させます。BBoxフォールバック時は上下左右それぞれへ15%広げます。高精度検出をONにすると、元画像全体に加えて1280px単位の分割推論を行います。推論解像度は`imgsz=1280`固定です。

自動処理後は、処理キューから画像を選んでProcessedタブを開きます。ブラシで追加範囲を塗り、必要に応じて消しゴムや元に戻すを使ってから「追加モザイクを保存」を押してください。ブラシ範囲は緑色で表示されます。

PDFが必要な場合は「処理完了時にPDFを自動作成」をONにします。手動補正後は「出力画像からPDFを作成」を押すと、出力フォルダ内の画像からPDFを作り直せます。画像はファイル名順で、1画像につき1ページになります。

## 使用モデル

推奨モデルは[01miku/anime-nsfw-segm-yolo26](https://huggingface.co/01miku/anime-nsfw-segm-yolo26)のXLモデルです。

- 使用ファイル: `nsfw-anime-xl-x1280.pt`
- 配置先: `models/nsfw-anime-xl-x1280.pt`
- モデルライセンス: MIT（配布元の最新情報を確認してください）

モデルは約135MBあり、GitHubの通常ファイル上限を超えるため、このリポジトリには含めません。`DOWNLOAD_RECOMMENDED_MODEL.bat`を実行するか、Hugging Faceから手動でダウンロードしてください。

`model.names`からクラス一覧を生成し、初期状態では`vagina`と`penis`だけをモザイク対象にします。`nipple / anus / pubic hair / female face / male face`は検出ログとDetection表示には含まれますが、初期状態では加工しません。

自動検出は完全ではありません。公開前には必ず出力画像を目視確認してください。

## CPU / GPU

- NVIDIA GPUあり: CUDA版PyTorchを自動導入し、GPUで推論します。
- NVIDIA GPUなし: CPU版PyTorchを導入し、そのまま利用できます。
- CPU利用時は、特に高精度検出をONにすると処理時間が長くなります。
- CUDAで問題が起きた場合は、安全のためCPUへ自動的に切り替えます。

## セットアップ

Python 3.10以上を用意し、PowerShellで次を実行します。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

または、エクスプローラーでこのフォルダを開き、PowerShellから`./setup.ps1`を実行します。完了後は`run_app.bat`で起動できます。

推奨モデルを自動取得する場合は、次をダブルクリックします。

```text
DOWNLOAD_RECOMMENDED_MODEL.bat
```

起動時にNVIDIA GPUが見つかり、PyTorchがCPU版だった場合は、CUDA 12.8版PyTorchの自動セットアップを開始します。GPUがなければCPU版を使用します。

## 起動

通常は`NSFW_Auto_Mosaic_Flow_START.bat`をダブルクリックしてください。初回は必要な環境を自動で準備します。

```powershell
python main.py
```

1. YOLOモデルファイルを選択します。
2. 入力フォルダと出力フォルダを選択します。
3. モデル定義に合う対象クラスを選択します。初期値は`vagina,penis`です。
4. 設定またはプリセットを選び、`処理開始`を押します。
5. プレビューと出力画像を目視確認します。

設定は終了時と変更時に`settings.json`へ保存されます。

## テスト

```powershell
python -m unittest discover -s tests -v
```

## EXE化

```powershell
pyinstaller --noconfirm --clean nsfw_auto_mosaic_flow.spec
```

生成物は`dist/NSFW Auto Mosaic Flow/NSFW Auto Mosaic Flow.exe`です。モデルは別途ダウンロードして`models`フォルダへ配置します。

## ライセンス

アプリ本体は[MIT License](LICENSE)で公開しています。推奨モデルは別プロジェクトです。モデルのライセンスと利用条件は配布元で確認してください。

## 注意事項

- モデルの検出精度とライセンスはユーザー側で確認してください。
- `.pt`は内部的にPythonオブジェクトを含むことがあります。出所を信頼できるモデルだけを読み込んでください。
- このアプリは学習機能を持ちません。
- Segmentation Maskを優先しますが、MaskのないDetectionモデルもBBoxフォールバックで利用できます。
- 入力画像は外部へ送信せず、ローカルで処理します。
- 読み込みや推論に失敗した画像は、可能な限り元画像を出力先へコピーします。
