"""
Chinese character converter using OpenCC for reliable Simplified to Traditional conversion.
Provides fast and accurate character-level conversion without semantic changes.
"""

import logging

logger = logging.getLogger(__name__)


class ChineseConverter:
    """
    Chinese character converter using OpenCC for reliable Simplified to Traditional conversion.
    """

    def __init__(self):
        self._opencc_converter = None
        self._opencc_available = False

        self._init_opencc()

    def _init_opencc(self):
        """Initialize OpenCC converter."""
        try:
            import opencc
            self._opencc_converter = opencc.OpenCC('s2t')  # Simplified to Traditional
            self._opencc_available = True
            logger.info("OpenCC converter initialized successfully")
        except ImportError:
            logger.warning("OpenCC not available. Install with: pip install opencc-python-reimplemented")
            self._opencc_available = False
        except Exception as e:
            logger.error(f"Failed to initialize OpenCC: {e}")
            self._opencc_available = False

    def convert_to_traditional(self, text: str) -> str:
        """Convert Simplified Chinese text to Traditional Chinese using OpenCC.
        
        Args:
            text: Simplified Chinese text
            
        Returns:
            Traditional Chinese text
        """
        if not text:
            return text

        if not self._opencc_available:
            logger.warning("OpenCC not available, falling back to basic conversion")
            return self._basic_conversion(text)

        try:
            converted_text = self._opencc_converter.convert(text)
            converted_count = sum(1 for s, t in zip(text, converted_text) if s != t)

            if converted_count > 0:
                logger.info(f"OpenCC converted {converted_count} characters from Simplified to Traditional")

            return converted_text
        except Exception as e:
            logger.error(f"OpenCC conversion failed: {e}, falling back to basic conversion")
            return self._basic_conversion(text)

    def _basic_conversion(self, text: str) -> str:
        """Basic fallback conversion using limited character mapping.
        
        Args:
            text: Simplified Chinese text
            
        Returns:
            Traditional Chinese text (basic conversion)
        """
        # Basic character mapping for fallback
        basic_mapping = {
            '这': '這', '个': '個', '为': '為', '们': '們', '来': '來',
            '时': '時', '对': '對', '会': '會', '发': '發', '国': '國',
            '电': '電', '话': '話', '计': '計', '算': '算', '机': '機',
            '网': '網', '络': '絡', '软': '軟', '统': '統', '术': '術',
            '说': '說', '过': '過', '还': '還', '没': '沒', '问': '問',
            '题': '題', '现': '現', '学': '學', '习': '習', '试': '試',
            '验': '驗', '脑': '腦', '爱': '愛', '觉': '覺', '开': '開',
            '兴': '興', '乐': '樂', '美': '美', '丽': '麗', '帅': '帥',
            '万': '萬', '亿': '億', '着': '著', '给': '給', '长': '長',
            '样': '樣', '经': '經', '总': '總', '应': '應', '该': '該',
            '让': '讓', '记': '記', '听': '聽', '见': '見', '买': '買',
            '卖': '賣', '钱': '錢', '块': '塊', '员': '員', '师': '師',
            '儿': '兒', '红': '紅', '绿': '綠', '蓝': '藍', '黄': '黃',
            '车': '車', '飞': '飛', '场': '場', '饭': '飯', '鱼': '魚',
            '鸡': '雞', '面': '麵', '气': '氣', '风': '風', '云': '雲',
            '阳': '陽'
        }

        converted_text = ""
        converted_count = 0

        for char in text:
            if char in basic_mapping:
                converted_text += basic_mapping[char]
                converted_count += 1
            else:
                converted_text += char

        if converted_count > 0:
            logger.info(f"Basic conversion: {converted_count} characters from Simplified to Traditional Chinese")

        return converted_text

# Global converter instance
_converter = None


def get_converter() -> ChineseConverter:
    """Get Chinese converter instance.
    
    Returns:
        ChineseConverter instance
    """
    global _converter
    if _converter is None:
        _converter = ChineseConverter()
    return _converter


def convert_to_traditional(text: str) -> str:
    """Convert Simplified Chinese text to Traditional Chinese using OpenCC.
    
    Args:
        text: Simplified Chinese text
        
    Returns:
        Traditional Chinese text
    """
    converter = get_converter()
    return converter.convert_to_traditional(text)
