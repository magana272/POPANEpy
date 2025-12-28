"""
Tests for emotion.dataloader.downloader module
"""
import os
import tempfile
import threading
import unittest
import zipfile
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from emotion.dataloader.downloader import POPANEDownloader


class TestPOPANEDownloaderInit(unittest.TestCase):
    """Test POPANEDownloader initialization"""

    def test_init_default_data_dir(self):
        """Test initialization with default data directory"""
        downloader = POPANEDownloader()
        self.assertEqual(downloader.data_dir, "./data/raw/")

    def test_init_custom_data_dir(self):
        """Test initialization with custom data directory"""
        downloader = POPANEDownloader(data_dir="/custom/path/")
        self.assertEqual(downloader.data_dir, "/custom/path/")

    def test_init_creates_session(self):
        """Test that initialization creates a requests session"""
        downloader = POPANEDownloader()
        self.assertIsNotNone(downloader._session)

    def test_init_lock_created(self):
        """Test that lock is created"""
        downloader = POPANEDownloader()
        self.assertIsInstance(downloader.lock, type(threading.Lock()))

    def test_init_download_state(self):
        """Test initial download state"""
        downloader = POPANEDownloader()
        self.assertFalse(downloader._downloads_completed)
        self.assertEqual(downloader._number_of_downloads, 0)
        self.assertEqual(downloader.completed_files, [])


class TestPOPANEDownloaderDataInfo(unittest.TestCase):
    """Test DATA_INFO structure"""

    def setUp(self):
        self.downloader = POPANEDownloader()

    def test_data_info_has_all_studies(self):
        """Test DATA_INFO contains all 7 studies plus meta"""
        for i in range(1, 8):
            self.assertIn(i, self.downloader.DATA_INFO)
        self.assertIn("meta", self.downloader.DATA_INFO)

    def test_data_info_structure(self):
        """Test each DATA_INFO entry has required keys"""
        for key, info in self.downloader.DATA_INFO.items():
            self.assertIn("uri", info)
            self.assertIn("info", info)
            self.assertIn("zipGB", info)
            self.assertIn("unzippedGB", info)

    def test_data_info_uris_are_valid(self):
        """Test URIs are valid URLs"""
        for key, info in self.downloader.DATA_INFO.items():
            self.assertTrue(info["uri"].startswith("https://"))


class TestPOPANEDownloaderSession(unittest.TestCase):
    """Test session creation"""

    def test_create_session_returns_session(self):
        """Test _create_session returns a requests Session"""
        import requests
        downloader = POPANEDownloader()
        session = downloader._create_session()
        self.assertIsInstance(session, requests.Session)

    def test_session_has_adapters(self):
        """Test session has HTTP adapters mounted"""
        downloader = POPANEDownloader()
        session = downloader._create_session()
        self.assertIn('https://', session.adapters)
        self.assertIn('http://', session.adapters)


class TestPOPANEDownloaderStatus(unittest.TestCase):
    """Test download status methods"""

    def setUp(self):
        self.downloader = POPANEDownloader()

    def test_download_complete_initial(self):
        """Test download_complete returns False initially"""
        self.assertFalse(self.downloader.download_complete())

    def test_number_of_downloads_initial(self):
        """Test number_of_downloads returns 0 initially"""
        self.assertEqual(self.downloader.number_of_downloads(), 0)

    def test_download_complete_after_update(self):
        """Test download_complete after updates"""
        self.downloader._downloads_completed = True
        self.assertTrue(self.downloader.download_complete())

    def test_number_of_downloads_after_update(self):
        """Test number_of_downloads after updates"""
        self.downloader._number_of_downloads = 5
        self.assertEqual(self.downloader.number_of_downloads(), 5)


class TestPOPANEDownloaderUnzip(unittest.TestCase):
    """Test unzip functionality"""

    def setUp(self):
        self.downloader = POPANEDownloader()
        self.test_dir = tempfile.mkdtemp()
        self.downloader.data_dir = self.test_dir

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_unzip_file(self):
        """Test unzipping a file"""
        # Create a test zip file
        zip_path = os.path.join(self.test_dir, "test.zip")
        test_content = b"test content"

        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test.txt", test_content)

        self.downloader.unzip_file(zip_path, self.test_dir)

        extracted_path = os.path.join(self.test_dir, "test.txt")
        self.assertTrue(os.path.exists(extracted_path))
        with open(extracted_path, 'rb') as f:
            self.assertEqual(f.read(), test_content)

    def test_unzip_files_empty_list(self):
        """Test unzip_files with empty completed_files"""
        self.downloader.completed_files = []
        self.downloader.unzip_files()  # Should not raise

    def test_unzip_files_with_non_zip(self):
        """Test unzip_files ignores non-zip files"""
        self.downloader.completed_files = ["/path/to/file.txt"]
        self.downloader.unzip_files()  # Should not raise

    def test_unzip_files_with_zip(self):
        """Test unzip_files extracts zip files"""
        zip_path = os.path.join(self.test_dir, "test.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("inner.txt", "content")

        self.downloader.completed_files = [zip_path]
        self.downloader.unzip_files()

        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "inner.txt")))

    def test_unzip_files_handles_exception(self):
        """Test unzip_files handles extraction errors gracefully"""
        self.downloader.completed_files = ["/nonexistent/path.zip"]
        # Should not raise, just print error
        self.downloader.unzip_files()


class TestPOPANEDownloaderUpdateDownloads(unittest.TestCase):
    """Test _update_number_of_downloads method"""

    def setUp(self):
        self.downloader = POPANEDownloader()

    def test_update_increments_count(self):
        """Test update increments download count"""
        self.downloader._update_number_of_downloads("test.txt", "/path/test.txt")
        self.assertEqual(self.downloader._number_of_downloads, 1)

    def test_update_adds_zip_to_completed(self):
        """Test update adds zip files to completed_files"""
        self.downloader._update_number_of_downloads("test.zip", "/path/test.zip")
        self.assertIn("/path/test.zip", self.downloader.completed_files)

    def test_update_does_not_add_non_zip(self):
        """Test update does not add non-zip files to completed_files"""
        self.downloader._update_number_of_downloads("test.txt", "/path/test.txt")
        self.assertNotIn("/path/test.txt", self.downloader.completed_files)

    def test_update_sets_completed_at_8(self):
        """Test downloads_completed is set True at 8 downloads"""
        for i in range(8):
            self.downloader._update_number_of_downloads(f"test{i}.txt", f"/path/test{i}.txt")
        self.assertTrue(self.downloader._downloads_completed)

    def test_update_not_completed_at_7(self):
        """Test downloads_completed is False at 7 downloads"""
        for i in range(7):
            self.downloader._update_number_of_downloads(f"test{i}.txt", f"/path/test{i}.txt")
        self.assertFalse(self.downloader._downloads_completed)


class TestPOPANEDownloaderDownloadData(unittest.TestCase):
    """Test download_data method with mocking"""

    def setUp(self):
        self.downloader = POPANEDownloader()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    @patch('builtins.input', return_value='n')
    def test_download_data_cancelled(self, mock_input):
        """Test download is cancelled when user says no"""
        self.downloader.download_data(directory=self.test_dir, studies=[1], confirm=True)
        # Should return early without downloading

    def test_download_data_no_confirm(self):
        """Test download without confirmation prompt"""
        with patch.object(self.downloader, '_build_ranges') as mock_build:
            with patch.object(self.downloader, 'unzip_files'):
                mock_build.return_value = []
                self.downloader.download_data(
                    directory=self.test_dir,
                    studies=[1],
                    confirm=False
                )

    def test_download_data_single_study_int(self):
        """Test download_data with single study as int"""
        with patch.object(self.downloader, '_build_ranges') as mock_build:
            with patch.object(self.downloader, 'unzip_files'):
                mock_build.return_value = []
                self.downloader.download_data(
                    directory=self.test_dir,
                    studies=1,  # Single int
                    confirm=False
                )

    def test_download_data_single_study_str(self):
        """Test download_data with single study as string"""
        with patch.object(self.downloader, '_build_ranges') as mock_build:
            with patch.object(self.downloader, 'unzip_files'):
                mock_build.return_value = []
                self.downloader.download_data(
                    directory=self.test_dir,
                    studies="meta",  # Single string
                    confirm=False
                )

    def test_download_data_creates_directory(self):
        """Test download_data creates directory if not exists"""
        new_dir = os.path.join(self.test_dir, "new_subdir")
        with patch.object(self.downloader, '_build_ranges') as mock_build:
            with patch.object(self.downloader, 'unzip_files'):
                mock_build.return_value = []
                self.downloader.download_data(
                    directory=new_dir,
                    studies=[1],
                    confirm=False
                )
        self.assertTrue(os.path.exists(new_dir))

    def test_download_data_uses_default_directory(self):
        """Test download_data uses data_dir when directory is None"""
        self.downloader.data_dir = self.test_dir
        with patch.object(self.downloader, '_build_ranges') as mock_build:
            with patch.object(self.downloader, 'unzip_files'):
                mock_build.return_value = []
                self.downloader.download_data(
                    directory=None,
                    studies=[1],
                    confirm=False
                )


class TestPOPANEDownloaderBuildRanges(unittest.TestCase):
    """Test _build_ranges method"""

    def setUp(self):
        self.downloader = POPANEDownloader()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_build_ranges_no_range_support(self):
        """Test _build_ranges raises when server doesn't support ranges"""
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "1000"}
        mock_response.raise_for_status = Mock()

        with patch.object(self.downloader._session, 'head', return_value=mock_response):
            with self.assertRaises(RuntimeError) as context:
                filepath = os.path.join(self.test_dir, "test.zip")
                self.downloader._build_ranges("http://example.com/file.zip", filepath, 4)
            self.assertIn("does not support range requests", str(context.exception))

    def test_build_ranges_creates_file(self):
        """Test _build_ranges creates sparse file"""
        mock_response = Mock()
        mock_response.headers = {"Accept-Ranges": "bytes", "Content-Length": "1000"}
        mock_response.raise_for_status = Mock()

        with patch.object(self.downloader._session, 'head', return_value=mock_response):
            filepath = os.path.join(self.test_dir, "test.zip")
            ranges = self.downloader._build_ranges("http://example.com/file.zip", filepath, 4)

            self.assertTrue(os.path.exists(filepath))
            self.assertEqual(os.path.getsize(filepath), 1000)

    def test_build_ranges_returns_correct_ranges(self):
        """Test _build_ranges returns correct byte ranges"""
        mock_response = Mock()
        mock_response.headers = {"Accept-Ranges": "bytes", "Content-Length": "1000"}
        mock_response.raise_for_status = Mock()

        with patch.object(self.downloader._session, 'head', return_value=mock_response):
            filepath = os.path.join(self.test_dir, "test.zip")
            ranges = self.downloader._build_ranges("http://example.com/file.zip", filepath, 4)

            self.assertEqual(len(ranges), 4)
            # Check ranges cover full file
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], 999)


class TestPOPANEDownloaderDownloadRange(unittest.TestCase):
    """Test _download_range method"""

    def setUp(self):
        self.downloader = POPANEDownloader()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_download_range_success(self):
        """Test successful range download"""
        # Create a file to write to
        filepath = os.path.join(self.test_dir, "test.zip")
        with open(filepath, 'wb') as f:
            f.write(b'\x00' * 100)

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_content = Mock(return_value=[b'test_content'])
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch.object(self.downloader._session, 'get', return_value=mock_response):
            result = self.downloader._download_range(
                "http://example.com/file.zip",
                filepath,
                0,
                50
            )
            self.assertEqual(result, (filepath, "test.zip"))

    def test_download_range_error(self):
        """Test _download_range handles errors"""
        filepath = os.path.join(self.test_dir, "test.zip")

        with patch.object(self.downloader._session, 'get', side_effect=Exception("Network error")):
            with self.assertRaises(Exception):
                self.downloader._download_range(
                    "http://example.com/file.zip",
                    filepath,
                    0,
                    50
                )


if __name__ == '__main__':
    unittest.main()
