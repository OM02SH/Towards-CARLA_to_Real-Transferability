from typing import Dict, List, Any, Optional, Tuple

class QAPair:
    """Represents a single Question-Answer pair."""
    def __init__(self, qid: int, category: str, question: str, answer: str, layer: str, metadata: Dict[str, Any]):
        self.qid = qid
        self.category = category
        self.question = question
        self.answer = answer
        self.layer = layer
        self.metadata = metadata 

    def __repr__(self) -> str:
        return f"QAPair(qid={self.qid}, category='{self.category}', Q='{self.question[:30]}...', A='{self.answer[:30]}...')"


class ScenarioQA:
    """
    Parses and manages driving scenario question-answering data, 
    making every interaction retrievable as clear Question-Answer pairs.
    """
    def __init__(self, data_tuple: Tuple[int, int, Dict[str, int], Dict, Dict] = None, raw_qa_data: Dict[str, List[Dict[str, Any]]] = None):
        #self.scenario_id = data_tuple[0]
        #self.some_flag = data_tuple[1]
        #self.category_counts = data_tuple[2]
        #self.extra_dict = data_tuple[3]
        #self.ego_state = data_tuple[4]
        
        self.qa_pairs: List[QAPair] = []
        self._parse_raw_qa(raw_qa_data)

    def _parse_raw_qa(self, raw_qa_data: Dict[str, List[Dict[str, Any]]]):
        """Pairs consecutive 'human' (question) and 'rule' (answer) items together."""
        for category, items in raw_qa_data.items():
            grouped_by_qid: Dict[int, Dict[str, Any]] = {}
            for item in items:
                qid = item['qid']
                role = item['from']
                
                if qid not in grouped_by_qid:
                    grouped_by_qid[qid] = {'human': None, 'rule': None}
                
                grouped_by_qid[qid][role] = item

            for qid, roles in grouped_by_qid.items():
                human_item = roles.get('human')
                rule_item = roles.get('rule')
                
                if human_item and rule_item:
                    # Extract common/metadata fields
                    metadata = {
                        'chain': human_item.get('chain'),
                        'type': human_item.get('type'),
                        'object_id': human_item.get('object_id'),
                        'connection_up': human_item.get('connection_up'),
                        'connection_down': human_item.get('connection_down'),
                        'object_tags': human_item.get('object_tags')
                    }
                    
                    qa_pair = QAPair(
                        qid=qid,
                        category=category,
                        question=human_item['value'],
                        answer=rule_item['value'],
                        layer = human_item['layer'],
                        metadata=metadata
                    )
                    self.qa_pairs.append(qa_pair)

    def get_all_pairs(self) -> List[QAPair]:
        """Returns all QA pairs."""
        return self.qa_pairs

    def get_by_category(self, category: str) -> List[QAPair]:
        """Filters QA pairs by category (e.g., 'ego', 'roadlayout')."""
        return [qa for qa in self.qa_pairs if qa.category.lower() == category.lower()]

    def get_by_qid(self, qid: int) -> Optional[QAPair]:
        """Retrieves a single QA pair matching the unique QID."""
        for qa in self.qa_pairs:
            if qa.qid == qid:
                return qa
        return None
    
    def get_all_questions(self, separator: str = " ") -> str:
        return separator.join(qa.question for qa in self.qa_pairs)

    def reset(self):
        self.qa_pairs = []
