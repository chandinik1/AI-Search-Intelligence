# Small standalone smoke tests for the expected cleaning behavior.
import unittest
from bs4 import BeautifulSoup

class HtmlExtractionSmokeTests(unittest.TestCase):
    def test_html_structure(self):
        soup = BeautifulSoup('<header>Global menu</header><main><h1>Industrial Gateway</h1><p>MQTT-compatible industrial devices connect edge systems.</p></main>', 'html.parser')
        for tag in soup(['header', 'nav', 'footer']):
            tag.decompose()
        self.assertNotIn('Global menu', soup.get_text(' ', strip=True))
        self.assertIn('MQTT-compatible', soup.get_text(' ', strip=True))

if __name__ == '__main__':
    unittest.main()
