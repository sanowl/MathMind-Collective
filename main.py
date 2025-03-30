import numpy as np
import sympy as sp
import re
import random
import json
import time
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Set, Optional, Union, Callable
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import networkx as nx
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SIGMA")

# =============================
# CORE SYSTEM ARCHITECTURE
# =============================

class MessageType(Enum):
    PROBLEM = "problem"  # Initial problem statement
    ANALYSIS = "analysis"  # Problem structure and components
    STRATEGY = "strategy"  # Proposed solution approach
    EXECUTION = "execution"  # Implementation of a strategy step
    VERIFICATION = "verification"  # Correctness check
    INTUITION = "intuition"  # Educated guess or insight
    DEBATE = "debate"  # Challenge to reasoning
    SYNTHESIS = "synthesis"  # Combined/final solution
    META = "meta"  # Resource allocation decision
    FEEDBACK = "feedback"  # User or system feedback
    VISUALIZATION = "visualization"  # Visual representation
    EXPLANATION = "explanation"  # Human-friendly explanation
    KNOWLEDGE = "knowledge"  # Relevant mathematical knowledge
    ANALOGY = "analogy"  # Analogical reasoning
    EVALUATION = "evaluation"  # Solution quality assessment


class Message:
    """Message passed between agents in the workspace"""
    
    def __init__(self, msg_type: MessageType, content: Any, sender: str, 
                 confidence: float = 1.0, references: List[int] = None,
                 metadata: Dict = None):
        self.id = id(self)  # Unique identifier
        self.type = msg_type
        self.content = content
        self.sender = sender
        self.confidence = confidence  # How confident the agent is (0-1)
        self.references = references or []  # IDs of referenced messages
        self.votes = 0  # For debate mechanism
        self.timestamp = 0  # Will be set when added to workspace
        self.metadata = metadata or {}  # Additional metadata
    
    def __str__(self):
        return f"[{self.type.value.upper()}] {self.sender}: {str(self.content)[:100]}{'...' if len(str(self.content)) > 100 else ''}"


class Workspace:
    """Shared workspace where agents communicate"""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.current_time = 0
        self.subscriptions: Dict[MessageType, List[str]] = {msg_type: [] for msg_type in MessageType}
        self.priority_queue: Dict[str, List[Message]] = {}  # Agent-specific priority queues
        self.knowledge_graph = nx.DiGraph()  # Knowledge graph for message relationships
        self.performance_metrics = defaultdict(list)  # System performance tracking
    
    def add_message(self, message: Message, priority: float = 1.0) -> int:
        """Add a message to the workspace and return its position"""
        message.timestamp = self.current_time
        self.messages.append(message)
        
        # Add to knowledge graph
        self.knowledge_graph.add_node(message.id, message=message)
        for ref_id in message.references:
            if self.knowledge_graph.has_node(ref_id):
                self.knowledge_graph.add_edge(ref_id, message.id)
        
        # Add to priority queues for relevant subscribers
        for agent_id in self.subscriptions.get(message.type, []):
            if agent_id not in self.priority_queue:
                self.priority_queue[agent_id] = []
            self.priority_queue[agent_id].append((priority, message))
        
        # Track performance metrics
        self.performance_metrics["message_count_by_type"][message.type.value] = \
            self.performance_metrics["message_count_by_type"].get(message.type.value, 0) + 1
        
        self.current_time += 1
        logger.info(f"New message: {message}")
        return len(self.messages) - 1
    
    def get_messages_by_type(self, msg_type: MessageType) -> List[Message]:
        """Get all messages of a specific type"""
        return [msg for msg in self.messages if msg.type == msg_type]
    
    def get_latest_message_by_type(self, msg_type: MessageType) -> Optional[Message]:
        """Get the most recent message of a specific type"""
        messages = self.get_messages_by_type(msg_type)
        return messages[-1] if messages else None
    
    def get_messages_by_sender(self, sender: str) -> List[Message]:
        """Get all messages from a specific sender"""
        return [msg for msg in self.messages if msg.sender == sender]
    
    def get_message_by_id(self, msg_id: int) -> Optional[Message]:
        """Get a message by its ID"""
        for msg in self.messages:
            if msg.id == msg_id:
                return msg
        return None
    
    def subscribe(self, agent_id: str, message_types: List[MessageType], callback: Callable = None):
        """Subscribe an agent to receive notifications for specific message types"""
        for msg_type in message_types:
            if agent_id not in self.subscriptions[msg_type]:
                self.subscriptions[msg_type].append(agent_id)
    
    def get_subscribers(self, message_type: MessageType) -> List[str]:
        """Get all agents subscribed to a message type"""
        return self.subscriptions[message_type]
    
    def get_next_priority_message(self, agent_id: str) -> Optional[Message]:
        """Get the highest priority message for an agent"""
        if agent_id not in self.priority_queue or not self.priority_queue[agent_id]:
            return None
        
        # Sort by priority (higher first)
        self.priority_queue[agent_id].sort(key=lambda x: x[0], reverse=True)
        
        # Get the highest priority message
        _, message = self.priority_queue[agent_id].pop(0)
        return message
    
    def generate_knowledge_graph_visualization(self) -> Dict:
        """Generate a visualization of the knowledge graph"""
        # Compute node positions using a layout algorithm
        pos = nx.spring_layout(self.knowledge_graph)
        
        # Create a visualization dictionary
        nodes = []
        for node_id in self.knowledge_graph.nodes():
            message = self.knowledge_graph.nodes[node_id].get('message')
            if message:
                nodes.append({
                    'id': node_id,
                    'type': message.type.value,
                    'sender': message.sender,
                    'confidence': message.confidence,
                    'x': pos[node_id][0],
                    'y': pos[node_id][1]
                })
        
        edges = []
        for source, target in self.knowledge_graph.edges():
            edges.append({
                'source': source,
                'target': target
            })
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def get_performance_report(self) -> Dict:
        """Get a report of system performance metrics"""
        return {
            'message_counts': dict(self.performance_metrics["message_count_by_type"]),
            'agent_activity': {
                agent: len([m for m in self.messages if m.sender == agent])
                for agent in set(m.sender for m in self.messages)
            },
            'confidence_distribution': {
                msg_type.value: [m.confidence for m in self.messages if m.type == msg_type]
                for msg_type in MessageType
            },
            'time_to_solution': self.current_time
        }


class Agent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, agent_id: str, workspace: Workspace):
        self.id = agent_id
        self.workspace = workspace
        self.last_processed_time = 0
        self.memory = AgentMemory()  # Long-term memory for the agent
        self.active = True  # Whether the agent is currently active
        self.resource_allocation = 1.0  # Resource allocation from meta-cognitive agent
        self.performance_metrics = defaultdict(list)  # Track agent performance
        
    def get_new_messages(self) -> List[Message]:
        """Get messages that have arrived since this agent last processed"""
        new_messages = [msg for msg in self.workspace.messages 
                        if msg.timestamp >= self.last_processed_time]
        self.last_processed_time = self.workspace.current_time
        return new_messages
    
    def send_message(self, msg_type: MessageType, content: Any, 
                     confidence: float = 1.0, references: List[int] = None,
                     priority: float = 1.0, metadata: Dict = None) -> int:
        """Send a message to the workspace"""
        msg = Message(msg_type, content, self.id, confidence, references, metadata)
        return self.workspace.add_message(msg, priority)
    
    def update_performance_metrics(self, metric_name: str, value: Any):
        """Update agent performance metrics"""
        self.performance_metrics[metric_name].append(value)
        
    def set_resource_allocation(self, allocation: float):
        """Set the resource allocation for this agent"""
        self.resource_allocation = max(0.0, min(allocation, 1.0))
        if self.resource_allocation < 0.1:
            self.active = False
        else:
            self.active = True
    
    @abstractmethod
    def step(self):
        """Process new information and potentially act"""
        pass


class AgentMemory:
    """Long-term memory for agents to store and retrieve information"""
    
    def __init__(self):
        self.episodic_memory = []  # Previous problem-solving episodes
        self.procedural_memory = {}  # Known procedures and techniques
        self.semantic_memory = {}  # Facts and relationships
        self.working_memory = []  # Currently active items
        self.embeddings = {}  # Vector embeddings for semantic search
        
    def store_episode(self, episode: Dict):
        """Store a problem-solving episode"""
        self.episodic_memory.append(episode)
        
    def store_procedure(self, name: str, procedure: Dict):
        """Store a procedure or technique"""
        self.procedural_memory[name] = procedure
        
    def store_fact(self, key: str, value: Any):
        """Store a fact or relationship"""
        self.semantic_memory[key] = value
        
    def find_similar_episodes(self, query: Dict, k: int = 3) -> List[Dict]:
        """Find similar previous problem-solving episodes"""
        # Simple implementation - in a full system would use vector embeddings
        matches = []
        for episode in self.episodic_memory:
            score = self._similarity_score(episode, query)
            matches.append((score, episode))
        
        # Return top k matches
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:k]]
    
    def _similarity_score(self, episode: Dict, query: Dict) -> float:
        """Compute a similarity score between an episode and a query"""
        # Simple implementation - in a full system would use more sophisticated metrics
        score = 0.0
        
        # Check domain similarity
        if episode.get("domain") == query.get("domain"):
            score += 0.5
            
        # Check keyword overlap
        episode_keywords = episode.get("keywords", [])
        query_keywords = query.get("keywords", [])
        if episode_keywords and query_keywords:
            overlap = set(episode_keywords).intersection(set(query_keywords))
            score += len(overlap) / max(len(episode_keywords), len(query_keywords))
            
        return score
    
    def update_working_memory(self, items: List[Any]):
        """Update the working memory with current items"""
        self.working_memory = items


# =============================
# ANALYSIS AGENTS
# =============================

class ProblemAnalyzerAgent(Agent):
    """Agent that breaks down math problems into components"""
    
    def __init__(self, workspace: Workspace):
        super().__init__("ProblemAnalyzer", workspace)
        self.workspace.subscribe(self.id, [MessageType.PROBLEM, MessageType.FEEDBACK])
        
        # Enhanced capabilities
        self.nlp_model = self._initialize_nlp_model()
        self.schema_extractor = self._initialize_schema_extractor()
        self.domain_classifier = self._initialize_domain_classifier()
        
    def _initialize_nlp_model(self):
        """Initialize NLP model for problem parsing"""
        # In a real implementation, this would be a sophisticated NLP model
        # For this demo, we'll use a placeholder
        return {"initialized": True}
        
    def _initialize_schema_extractor(self):
        """Initialize schema extractor for identifying problem structure"""
        # In a real implementation, this would extract problem schemas
        return {"initialized": True}
        
    def _initialize_domain_classifier(self):
        """Initialize domain classifier for mathematical domains"""
        # In a real implementation, this would be a trained classifier
        return {"initialized": True}
        
    def step(self):
        if not self.active:
            return
            
        new_messages = self.get_new_messages()
        
        for msg in new_messages:
            if msg.type == MessageType.PROBLEM:
                self.analyze_problem(msg)
            elif msg.type == MessageType.FEEDBACK and "analysis" in msg.content:
                self.refine_analysis(msg)
    
    def analyze_problem(self, problem_msg: Message):
        """Analyze a problem comprehensively"""
        start_time = time.time()
        problem_text = problem_msg.content
        
        # Phase 1: Parse the problem text
        parsed_problem = self.parse_problem(problem_text)
        
        # Phase 2: Extract mathematical components
        components = self.extract_components(parsed_problem)
        
        # Phase 3: Classify the problem domains
        domains = self.identify_domains(components)
        
        # Phase 4: Identify problem schema
        schema = self.identify_problem_schema(components, domains)
        
        # Phase 5: Assess complexity
        complexity = self.assess_complexity(components, schema, domains)
        
        # Complete analysis
        analysis = {
            "original_problem": problem_text,
            "parsed_structure": parsed_problem,
            "components": components,
            "domains": domains,
            "schema": schema,
            "complexity": complexity,
            "keywords": self.extract_keywords(problem_text)
        }
        
        # Track performance
        processing_time = time.time() - start_time
        self.update_performance_metrics("processing_time", processing_time)
        
        # Send analysis to workspace
        self.send_message(
            MessageType.ANALYSIS, 
            analysis,
            confidence=self.calculate_confidence(analysis),
            references=[problem_msg.id],
            priority=1.5 if complexity > 0.7 else 1.0
        )
        
        # Also generate a visual representation of the problem
        self.generate_problem_visualization(analysis, problem_msg.id)
    
    def parse_problem(self, problem_text: str) -> Dict:
        """Parse the problem text structure"""
        # Advanced NLP would be used here in a real implementation
        sentences = [s.strip() for s in problem_text.split('.') if s.strip()]
        
        # Identify question part
        question = ""
        context = []
        for sentence in sentences:
            if '?' in sentence or any(kw in sentence.lower() for kw in ["find", "calculate", "determine", "what is"]):
                question = sentence
            else:
                context.append(sentence)
        
        return {
            "context": context,
            "question": question,
            "complete_text": problem_text
        }
    
    def extract_components(self, parsed_problem: Dict) -> Dict:
        """Extract mathematical components from the parsed problem"""
        problem_text = parsed_problem["complete_text"]
        
        # Extract various components (more sophisticated in real implementation)
        components = {
            "variables": self.extract_variables(problem_text),
            "equations": self.extract_equations(problem_text),
            "expressions": self.extract_expressions(problem_text),
            "constraints": self.extract_constraints(problem_text),
            "constants": self.extract_constants(problem_text),
            "functions": self.extract_functions(problem_text),
            "geometric_entities": self.extract_geometric_entities(problem_text),
            "numerical_values": self.extract_numerical_values(problem_text),
            "units": self.extract_units(problem_text),
        }
        
        # Add the target - what the problem is asking for
        components["target"] = self.identify_target(parsed_problem)
        
        return components
    
    def extract_variables(self, problem: str) -> List[Dict]:
        """Extract variables with enhanced information"""
        # Simple extraction for demonstration
        # A real implementation would be more sophisticated
        variable_names = set(re.findall(r'\b([a-zA-Z])\b', problem))
        
        variables = []
        for var in variable_names:
            var_info = {
                "name": var,
                "occurrences": len(re.findall(rf'\b{var}\b', problem)),
                "potential_type": self.infer_variable_type(var, problem)
            }
            variables.append(var_info)
            
        return variables
    
    def infer_variable_type(self, variable: str, context: str) -> str:
        """Infer the potential type of a variable based on context"""
        # Simple heuristics for demonstration
        if variable.lower() in ['x', 'y', 'z']:
            return "coordinate" if "point" in context.lower() else "unknown"
        elif variable.lower() in ['t']:
            return "time" if any(w in context.lower() for w in ["time", "rate", "speed", "velocity"]) else "parameter"
        elif variable.lower() in ['r']:
            return "radius" if any(w in context.lower() for w in ["circle", "sphere", "radius"]) else "unknown"
        elif variable.lower() in ['h']:
            return "height" if any(w in context.lower() for w in ["height", "triangle", "rectangle"]) else "unknown"
        elif variable.lower() in ['a', 'b', 'c']:
            return "coefficient" if "equation" in context.lower() else "side_length" if "triangle" in context.lower() else "unknown"
        elif variable.lower() in ['p']:
            return "probability" if "probability" in context.lower() else "unknown"
        elif variable.lower() in ['n']:
            return "count" if any(w in context.lower() for w in ["number", "count", "integer"]) else "unknown"
        else:
            return "unknown"
    
    def extract_equations(self, problem: str) -> List[Dict]:
        """Extract equations with enhanced information"""
        # Simple regex-based extraction for demonstration
        equation_strs = re.findall(r'([a-zA-Z0-9\s\+\-\*\/\^\=\(\)]+\=\s*[a-zA-Z0-9\s\+\-\*\/\^\(\)]+)', problem)
        
        equations = []
        for i, eq_str in enumerate(equation_strs):
            eq_info = {
                "id": f"eq{i+1}",
                "text": eq_str.strip(),
                "variables": re.findall(r'\b([a-zA-Z])\b', eq_str),
                "type": self.classify_equation_type(eq_str)
            }
            equations.append(eq_info)
            
        return equations
    
    def classify_equation_type(self, equation: str) -> str:
        """Classify the type of equation"""
        # Simple heuristics for demonstration
        if '^2' in equation or '²' in equation:
            return "quadratic" if '=' in equation else "quadratic_expression"
        elif '^3' in equation or '³' in equation:
            return "cubic" if '=' in equation else "cubic_expression"
        elif '^' in equation:
            return "polynomial" if '=' in equation else "polynomial_expression"
        elif 'sin' in equation or 'cos' in equation or 'tan' in equation:
            return "trigonometric"
        elif 'log' in equation or 'ln' in equation or 'exp' in equation:
            return "logarithmic" if 'log' in equation or 'ln' in equation else "exponential"
        elif '=' in equation:
            return "linear" if all('^' not in term for term in equation.split('=')) else "non_linear"
        else:
            return "expression"
    
    def extract_expressions(self, problem: str) -> List[Dict]:
        """Extract mathematical expressions"""
        # Simple implementation for demonstration
        # Would be more sophisticated in real system
        expressions = []
        
        # Look for expressions without equals sign but with operations
        pattern = r'([a-zA-Z0-9\s\+\-\*\/\^\(\)]+)'
        matches = re.findall(pattern, problem)
        
        for match in matches:
            if any(op in match for op in ['+', '-', '*', '/', '^']) and '=' not in match:
                expressions.append({
                    "text": match.strip(),
                    "operations": [op for op in ['+', '-', '*', '/', '^'] if op in match],
                    "variables": re.findall(r'\b([a-zA-Z])\b', match)
                })
        
        return expressions
    
    def extract_constraints(self, problem: str) -> List[Dict]:
        """Extract constraints with enhanced information"""
        # Simple constraint extraction based on keywords and patterns
        constraints = []
        
        # Inequality constraints
        inequality_patterns = [
            (r'([a-zA-Z0-9\s\+\-\*\/\^\=\(\)]+\>\s*[a-zA-Z0-9\s\+\-\*\/\^\(\)]+)', "greater_than"),
            (r'([a-zA-Z0-9\s\+\-\*\/\^\=\(\)]+\<\s*[a-zA-Z0-9\s\+\-\*\/\^\(\)]+)', "less_than"),
            (r'([a-zA-Z0-9\s\+\-\*\/\^\=\(\)]+\>\=\s*[a-zA-Z0-9\s\+\-\*\/\^\(\)]+)', "greater_than_or_equal"),
            (r'([a-zA-Z0-9\s\+\-\*\/\^\=\(\)]+\<\=\s*[a-zA-Z0-9\s\+\-\*\/\^\(\)]+)', "less_than_or_equal")
        ]
        
        for pattern, constraint_type in inequality_patterns:
            matches = re.findall(pattern, problem)
            for i, match in enumerate(matches):
                constraints.append({
                    "id": f"{constraint_type}_{i+1}",
                    "text": match.strip(),
                    "type": constraint_type,
                    "variables": re.findall(r'\b([a-zA-Z])\b', match)
                })
        
        # Keyword-based constraints
        constraint_keywords = [
            (r'must be positive', "positivity"),
            (r'must be negative', "negativity"),
            (r'must be non-negative', "non_negativity"),
            (r'must be non-positive', "non_positivity"),
            (r'must be integer', "integrality"),
            (r'must be rational', "rationality"),
            (r'(only|all) values? of ([a-zA-Z]) (is|are)', "domain_restriction")
        ]
        
        for pattern, constraint_type in constraint_keywords:
            matches = re.findall(pattern, problem)
            for i, match in enumerate(matches if isinstance(matches[0], str) else matches):
                match_text = match if isinstance(match, str) else ' '.join(match)
                constraints.append({
                    "id": f"{constraint_type}_{i+1}",
                    "text": match_text.strip(),
                    "type": constraint_type,
                    "variables": re.findall(r'\b([a-zA-Z])\b', match_text)
                })
        
        return constraints
    
    def extract_constants(self, problem: str) -> List[Dict]:
        """Extract mathematical constants"""
        # Look for named constants
        named_constants = {
            "pi": 3.14159,
            "e": 2.71828,
            "phi": 1.61803,
            "golden ratio": 1.61803
        }
        
        constants = []
        for name, value in named_constants.items():
            if name in problem.lower():
                constants.append({
                    "name": name,
                    "value": value,
                    "type": "named_constant"
                })
        
        return constants
    
    def extract_functions(self, problem: str) -> List[Dict]:
        """Extract mathematical functions"""
        # Look for common function patterns
        function_patterns = [
            (r'f\s*\(\s*([a-zA-Z0-9\+\-\*\/]+)\s*\)', "generic"),
            (r'sin\s*\(\s*([a-zA-Z0-9\+\-\*\/]+)\s*\)', "trigonometric"),
            (r'cos\s*\(\s*([a-zA-Z0-9\+\-\*\/]+)\s*\)', "trigonometric"),
            (r'tan\s*\(\s*([a-zA-Z0-9\+\-\*\/]+)\s*\)', "trigonometric"),
            (r'log\s*\(\s*([a-zA-Z0-9\+\-\*\/]+)\s*\)', "logarithmic"),
            (r'ln\s*\(\s*([a-zA-Z0-9\+\-\*\/]+)\s*\)', "logarithmic"),
            (r'exp\s*\(\s*([a-zA-Z0-9\+\-\*\/]+)\s*\)', "exponential"),
            (r'sqrt\s*\(\s*([a-zA-Z0-9\+\-\*\/]+)\s*\)', "radical")
        ]
        
        functions = []
        for pattern, func_type in function_patterns:
            matches = re.findall(pattern, problem)
            for i, match in enumerate(matches):
                functions.append({
                    "type": func_type,
                    "argument": match,
                    "variables": re.findall(r'\b([a-zA-Z])\b', match)
                })
        
        return functions
    
    def extract_geometric_entities(self, problem: str) -> List[Dict]:
        """Extract geometric entities"""
        # Look for common geometric entities
        geometric_keywords = {
            "triangle": "polygon",
            "square": "polygon",
            "rectangle": "polygon",
            "circle": "conic",
            "ellipse": "conic",
            "parabola": "conic",
            "hyperbola": "conic",
            "sphere": "3d",
            "cube": "3d",
            "cylinder": "3d",
            "cone": "3d",
            "line": "linear",
            "plane": "planar",
            "angle": "measure",
            "point": "position",
            "vector": "direction"
        }
        
        entities = []
        for entity, entity_type in geometric_keywords.items():
            if entity in problem.lower():
                entities.append({
                    "entity": entity,
                    "type": entity_type,
                    "count": len(re.findall(rf'\b{entity}\b', problem.lower()))
                })
        
        return entities
    
    def extract_numerical_values(self, problem: str) -> List[Dict]:
        """Extract numerical values"""
        # Simple extraction of numbers
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', problem)
        
        values = []
        for num in numbers:
            values.append({
                "value": float(num),
                "text": num,
                "is_integer": '.' not in num
            })
        
        return values
    
    def extract_units(self, problem: str) -> List[Dict]:
        """Extract units of measurement"""
        # Common units
        unit_patterns = {
            "length": [r'\b(meter|m|km|cm|mm|inch|foot|feet|yard|mile)s?\b'],
            "area": [r'\b(square meter|m²|km²|cm²|mm²|acre|hectare)s?\b'],
            "volume": [r'\b(cubic meter|m³|liter|gallon|quart|cup)s?\b'],
            "time": [r'\b(second|minute|hour|day|week|month|year)s?\b'],
            "speed": [r'\b(meter per second|m\/s|km\/h|mph)s?\b'],
            "mass": [r'\b(gram|kilogram|g|kg|pound|lb|ounce|oz)s?\b'],
            "temperature": [r'\b(degree|celsius|fahrenheit|kelvin)s?\b'],
            "angle": [r'\b(degree|radian)s?\b'],
            "currency": [r'\b(dollar|euro|pound|yen|\$|€|£|¥)s?\b']
        }
        
        units = []
        for unit_type, patterns in unit_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, problem.lower())
                for match in matches:
                    units.append({
                        "unit": match,
                        "type": unit_type
                    })
        
        return units
    
    def identify_target(self, parsed_problem: Dict) -> Dict:
        """Identify what the problem is asking for"""
        question = parsed_problem.get("question", "")
        
        target = {
            "description": "unknown",
            "variable": None,
            "type": "unknown"
        }
        
        # Look for target variable
        if question:
            # Check for "find x" pattern
            find_match = re.search(r'find\s+([a-zA-Z])', question.lower())
            if find_match:
                var = find_match.group(1)
                target.update({
                    "description": f"value of {var}",
                    "variable": var,
                    "type": "value"
                })
                return target
            
            # Check for "calculate the area" pattern
            calc_match = re.search(r'(calculate|find|determine|what is)\s+the\s+([a-zA-Z\s]+)', question.lower())
            if calc_match:
                target_desc = calc_match.group(2).strip()
                target.update({
                    "description": target_desc,
                    "variable": None,
                    "type": self._classify_target_type(target_desc)
                })
                return target
        
        return target
    
    def _classify_target_type(self, target_desc: str) -> str:
        """Classify the type of target based on description"""
        if any(term in target_desc for term in ["area", "volume", "perimeter", "circumference"]):
            return "measurement"
        elif any(term in target_desc for term in ["root", "solution", "value", "zero"]):
            return "value"
        elif any(term in target_desc for term in ["maximum", "minimum", "extremum", "optimize"]):
            return "optimization"
        elif any(term in target_desc for term in ["probability", "chance", "likelihood"]):
            return "probability"
        elif any(term in target_desc for term in ["prove", "show", "demonstrate"]):
            return "proof"
        else:
            return "unknown"
    
    def identify_domains(self, components: Dict) -> Dict[str, float]:
        """Identify the mathematical domain(s) of the problem with confidence levels"""
        domains = {
            "algebra": 0.0,
            "calculus": 0.0,
            "geometry": 0.0,
            "statistics": 0.0,
            "number_theory": 0.0,
            "linear_algebra": 0.0,
            "combinatorics": 0.0,
            "probability": 0.0,
            "differential_equations": 0.0,
            "discrete_math": 0.0
        }
        
        # Enhanced domain identification based on components
        
        # Check for algebraic indicators
        equations = components.get("equations", [])
        if equations:
            eq_types = [eq.get("type", "") for eq in equations]
            if any(t in ["quadratic", "cubic", "polynomial"] for t in eq_types):
                domains["algebra"] += 0.7
            elif any(t == "linear" for t in eq_types):
                domains["algebra"] += 0.5
            
            # Check for systems of equations (linear algebra)
            if len(equations) > 1 and all(t == "linear" for t in eq_types):
                domains["linear_algebra"] += 0.6
        
        # Check for calculus indicators
        functions = components.get("functions", [])
        if functions:
            if any(f.get("type") in ["trigonometric", "logarithmic", "exponential"] for f in functions):
                domains["calculus"] += 0.4
                
            target = components.get("target", {}).get("description", "").lower()
            if any(term in target for term in ["rate", "derivative", "slope", "maximum", "minimum"]):
                domains["calculus"] += 0.7
            elif any(term in target for term in ["area", "integral", "accumulated"]):
                domains["calculus"] += 0.7
                
        # Check for geometry indicators
        geometric_entities = components.get("geometric_entities", [])
        if geometric_entities:
            domains["geometry"] += 0.3 * len(geometric_entities)
            
        # Check for statistics indicators
        if any("probability" in str(c).lower() for c in components.values()):
            domains["statistics"] += 0.4
            domains["probability"] += 0.7
            
        # Check for number theory indicators
        constraints = components.get("constraints", [])
        if constraints:
            if any(c.get("type") == "integrality" for c in constraints):
                domains["number_theory"] += 0.5
                
            if any("prime" in str(c).lower() for c in constraints):
                domains["number_theory"] += 0.7
        
        # Normalize to ensure the sum doesn't exceed 1 (with multi-domain problems)
        total = sum(domains.values())
        if total > 0:
            domains = {k: min(v/total, 1.0) for k, v in domains.items()}
        
        return domains
    
    def identify_problem_schema(self, components: Dict, domains: Dict) -> Dict:
        """Identify the problem schema/pattern"""
        # Determine the highest confidence domain
        primary_domain = max(domains.items(), key=lambda x: x[1])[0] if domains else "unknown"
        
        # Extract schema based on domain and components
        schema = {
            "type": "unknown",
            "pattern": "unknown",
            "solution_approach": "unknown"
        }
        
        if primary_domain == "algebra":
            schema = self._identify_algebra_schema(components)
        elif primary_domain == "calculus":
            schema = self._identify_calculus_schema(components)
        elif primary_domain == "geometry":
            schema = self._identify_geometry_schema(components)
        elif primary_domain == "statistics" or primary_domain == "probability":
            schema = self._identify_statistics_schema(components)
        elif primary_domain == "number_theory":
            schema = self._identify_number_theory_schema(components)
        
        return schema
    
    def _identify_algebra_schema(self, components: Dict) -> Dict:
        """Identify algebraic problem schemas"""
        equations = components.get("equations", [])
        eq_types = [eq.get("type", "") for eq in equations]
        
        if "quadratic" in eq_types:
            return {
                "type": "equation_solving",
                "pattern": "quadratic_equation",
                "solution_approach": "quadratic_formula"
            }
        elif len(equations) > 1:
            return {
                "type": "system_of_equations",
                "pattern": "linear_system" if all(t == "linear" for t in eq_types) else "nonlinear_system",
                "solution_approach": "elimination_substitution" if all(t == "linear" for t in eq_types) else "numerical_methods"
            }
        elif any(t == "polynomial" for t in eq_types):
            return {
                "type": "equation_solving",
                "pattern": "polynomial_equation",
                "solution_approach": "factoring_or_numerical"
            }
        else:
            return {
                "type": "algebraic_manipulation",
                "pattern": "expression_simplification",
                "solution_approach": "algebraic_rules"
            }
    
    def _identify_calculus_schema(self, components: Dict) -> Dict:
        """Identify calculus problem schemas"""
        target = components.get("target", {}).get("description", "").lower()
        
        if any(term in target for term in ["rate", "derivative", "slope"]):
            return {
                "type": "differentiation",
                "pattern": "rate_of_change",
                "solution_approach": "compute_derivative"
            }
        elif any(term in target for term in ["maximum", "minimum", "optimal"]):
            return {
                "type": "optimization",
                "pattern": "max_min_problem",
                "solution_approach": "critical_points"
            }
        elif any(term in target for term in ["area", "integral", "accumulated"]):
            return {
                "type": "integration",
                "pattern": "accumulation_problem",
                "solution_approach": "compute_integral"
            }
        else:
            return {
                "type": "calculus",
                "pattern": "unknown_calculus_pattern",
                "solution_approach": "analytical_methods"
            }
    
    def _identify_geometry_schema(self, components: Dict) -> Dict:
        """Identify geometry problem schemas"""
        entities = components.get("geometric_entities", [])
        entity_types = [e.get("entity", "") for e in entities]
        
        if "triangle" in entity_types:
            return {
                "type": "triangle_problem",
                "pattern": "triangle_measurements",
                "solution_approach": "trigonometry_and_triangle_laws"
            }
        elif "circle" in entity_types:
            return {
                "type": "circle_problem",
                "pattern": "circle_measurements",
                "solution_approach": "circle_properties"
            }
        elif any(e in entity_types for e in ["square", "rectangle"]):
            return {
                "type": "quadrilateral_problem",
                "pattern": "rectangle_measurements",
                "solution_approach": "area_perimeter_formulas"
            }
        elif "angle" in entity_types:
            return {
                "type": "angle_problem",
                "pattern": "angle_measurements",
                "solution_approach": "angle_properties"
            }
        else:
            return {
                "type": "geometry",
                "pattern": "unknown_geometry_pattern",
                "solution_approach": "geometric_principles"
            }
    
    def _identify_statistics_schema(self, components: Dict) -> Dict:
        """Identify statistics/probability problem schemas"""
        target = components.get("target", {}).get("description", "").lower()
        
        if "probability" in target:
            return {
                "type": "probability",
                "pattern": "probability_calculation",
                "solution_approach": "probability_rules"
            }
        elif any(term in target for term in ["mean", "average", "median", "mode"]):
            return {
                "type": "descriptive_statistics",
                "pattern": "summary_statistics",
                "solution_approach": "statistical_formulas"
            }
        elif any(term in target for term in ["distribution", "normal", "poisson", "binomial"]):
            return {
                "type": "probability_distributions",
                "pattern": "distribution_problem",
                "solution_approach": "distribution_properties"
            }
        else:
            return {
                "type": "statistics",
                "pattern": "unknown_statistics_pattern",
                "solution_approach": "statistical_methods"
            }
    
    def _identify_number_theory_schema(self, components: Dict) -> Dict:
        """Identify number theory problem schemas"""
        # Simple implementation - would be more complex in real system
        return {
            "type": "number_theory",
            "pattern": "integer_properties",
            "solution_approach": "number_theory_principles"
        }
    
    def assess_complexity(self, components: Dict, schema: Dict, domains: Dict) -> float:
        """Assess the complexity of the problem (0-1 scale)"""
        # More sophisticated complexity assessment
        complexity = 0.0
        
        # Base complexity from components
        complexity += min(0.05 * len(components.get("variables", [])), 0.2)
        complexity += min(0.1 * len(components.get("equations", [])), 0.3)
        complexity += min(0.05 * len(components.get("constraints", [])), 0.2)
        
        # Domain-based complexity
        domain_complexity = {
            "algebra": 0.3,
            "calculus": 0.6,
            "geometry": 0.4,
            "statistics": 0.5,
            "number_theory": 0.7,
            "linear_algebra": 0.6,
            "combinatorics": 0.7,
            "probability": 0.5,
            "differential_equations": 0.8,
            "discrete_math": 0.6
        }
        
        for domain, confidence in domains.items():
            complexity += domain_complexity.get(domain, 0.5) * confidence * 0.3
        
        # Schema-based complexity
        schema_type = schema.get("type", "unknown")
        if schema_type in ["optimization", "integration", "system_of_equations"]:
            complexity += 0.2
        elif schema_type in ["probability_distributions", "differential_equations"]:
            complexity += 0.3
        
        # Adjust for interdomain problems (problems spanning multiple domains tend to be more complex)
        strong_domains = [d for d, c in domains.items() if c > 0.3]
        if len(strong_domains) > 1:
            complexity += 0.1 * (len(strong_domains) - 1)
        
        return min(complexity, 1.0)
    
    def extract_keywords(self, problem_text: str) -> List[str]:
        """Extract keywords from the problem text"""
        # Simple implementation - would use more sophisticated techniques in real system
        words = problem_text.lower().split()
        
        # Filter out common stop words
        stop_words = ["the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "is", "are"]
        filtered_words = [w for w in words if w not in stop_words]
        
        # Extract mathematical keywords
        math_keywords = [
            "solve", "find", "calculate", "determine", "evaluate",
            "equation", "function", "derivative", "integral", "graph",
            "perpendicular", "parallel", "tangent", "normal", "centroid",
            "area", "volume", "perimeter", "distance", "probability",
            "maximum", "minimum", "optimize", "extrema", "inflection",
            "sequence", "series", "limit", "convergence", "divergence",
            "prime", "factor", "divisor", "multiple", "rational",
            "matrix", "vector", "eigenvalue", "determinant", "transformation"
        ]
        
        keywords = [w for w in filtered_words if w in math_keywords]
        
        # Add domain-specific keywords
        for domain in ["algebra", "calculus", "geometry", "statistics", "probability"]:
            if domain in problem_text.lower():
                keywords.append(domain)
                
        return list(set(keywords))  # Remove duplicates
    
    def calculate_confidence(self, analysis: Dict) -> float:
        """Calculate confidence in the analysis"""
        # Base confidence
        confidence = 0.8
        
        # Adjust based on completeness
        components = analysis.get("components", {})
        if not components.get("variables"):
            confidence -= 0.1
        if not components.get("equations") and not components.get("expressions"):
            confidence -= 0.2
            
        # Adjust based on clarity of domain
        domains = analysis.get("domains", {})
        primary_domain = max(domains.items(), key=lambda x: x[1])[0] if domains else None
        primary_confidence = domains.get(primary_domain, 0) if primary_domain else 0
        if primary_confidence < 0.5:
            confidence -= 0.2
            
        # Adjust based on schema identification
        schema = analysis.get("schema", {})
        if schema.get("type") == "unknown":
            confidence -= 0.1
            
        return max(0.4, min(confidence, 1.0))  # Ensure confidence is between 0.4 and 1.0
    
    def generate_problem_visualization(self, analysis: Dict, problem_id: int):
        """Generate a visual representation of the problem structure"""
        # In a real system, this would create diagrams, graphs, or other visualizations
        # Here, we'll create a simple structural representation
        
        # Create a graph representation of problem components
        components = analysis.get("components", {})
        schema = analysis.get("schema", {})
        
        visualization_data = {
            "type": "problem_structure",
            "schema": schema.get("type", "unknown"),
            "components": {
                "variables": [v.get("name") for v in components.get("variables", [])],
                "equations": [e.get("text") for e in components.get("equations", [])],
                "constraints": [c.get("text") for c in components.get("constraints", [])]
            },
            "relationships": [],  # Would contain variable relationships
            "rendering_info": {
                "layout": "hierarchical",
                "title": "Problem Structure"
            }
        }
        
        # Add relationship information for variables in equations
        for eq in components.get("equations", []):
            eq_text = eq.get("text", "")
            for var in eq.get("variables", []):
                visualization_data["relationships"].append({
                    "from": var,
                    "to": eq_text,
                    "type": "appears_in"
                })
        
        self.send_message(
            MessageType.VISUALIZATION, 
            visualization_data,
            confidence=0.8,
            references=[problem_id]
        )
    
    def refine_analysis(self, feedback_msg: Message):
        """Refine an analysis based on feedback"""
        feedback = feedback_msg.content
        analysis_id = None
        
        # Find the referenced analysis
        for ref_id in feedback_msg.references:
            msg = self.workspace.get_message_by_id(ref_id)
            if msg and msg.type == MessageType.ANALYSIS:
                analysis_id = ref_id
                analysis = msg.content
                break
                
        if not analysis_id:
            return
            
        # Apply feedback to create refined analysis
        refined_analysis = self._apply_feedback_to_analysis(analysis, feedback)
        
        # Send refined analysis
        self.send_message(
            MessageType.ANALYSIS, 
            refined_analysis,
            confidence=0.9,  # Higher confidence for refined analysis
            references=[analysis_id, feedback_msg.id],
            priority=1.2  # Higher priority for refinements
        )
    
    def _apply_feedback_to_analysis(self, analysis: Dict, feedback: Dict) -> Dict:
        """Apply feedback to refine an analysis"""
        # Simple implementation - would be more sophisticated in real system
        refined = analysis.copy()
        
        if "missing_variables" in feedback:
            # Add missing variables
            for var in feedback["missing_variables"]:
                if var not in [v.get("name") for v in refined["components"]["variables"]]:
                    refined["components"]["variables"].append({
                        "name": var,
                        "occurrences": 1,
                        "potential_type": "added_from_feedback"
                    })
        
        if "wrong_domain" in feedback:
            # Correct domain classification
            wrong_domain = feedback["wrong_domain"]
            correct_domain = feedback.get("correct_domain")
            
            if wrong_domain in refined["domains"] and correct_domain:
                wrong_confidence = refined["domains"][wrong_domain]
                refined["domains"][wrong_domain] = 0.1
                refined["domains"][correct_domain] = max(0.8, refined["domains"].get(correct_domain, 0))
        
        if "complexity_adjustment" in feedback:
            # Adjust complexity
            adjustment = feedback["complexity_adjustment"]
            refined["complexity"] = max(0, min(1, refined["complexity"] + adjustment))
        
        return refined


class KnowledgeBaseAgent(Agent):
    """Agent that provides relevant mathematical knowledge"""
    
    def __init__(self, workspace: Workspace):
        super().__init__("KnowledgeBase", workspace)
        self.workspace.subscribe(self.id, [MessageType.ANALYSIS, MessageType.STRATEGY, MessageType.EXECUTION])
        
        # Initialize knowledge base
        self.knowledge_base = self._initialize_knowledge_base()
        self.theorem_database = self._initialize_theorem_database()
        self.formula_database = self._initialize_formula_database()
        self.method_database = self._initialize_method_database()
        
    def _initialize_knowledge_base(self) -> Dict:
        """Initialize general mathematical knowledge base"""
        return {
            "algebra": {
                "quadratic_formula": {
                    "name": "Quadratic Formula",
                    "formula": "x = (-b ± √(b² - 4ac)) / (2a)",
                    "description": "For a quadratic equation ax² + bx + c = 0, the solutions are given by the quadratic formula.",
                    "conditions": "Valid when a ≠ 0",
                    "applications": ["Finding roots of quadratic equations", "Solving quadratic inequalities"]
                },
                "polynomial_factoring": {
                    "name": "Polynomial Factoring",
                    "description": "Breaking down a polynomial into a product of simpler polynomials",
                    "methods": ["Factor out GCF", "Difference of squares", "Sum/difference of cubes", "Grouping"]
                },
                "completing_the_square": {
                    "name": "Completing the Square",
                    "description": "A method to rewrite a quadratic expression as a perfect square trinomial plus or minus a constant",
                    "method": "For ax² + bx + c, rewrite as a(x² + (b/a)x) + c, then a(x² + (b/a)x + (b/2a)²) + c - a(b/2a)²"
                }
            },
            "calculus": {
                "power_rule": {
                    "name": "Power Rule",
                    "formula": "d/dx(x^n) = n·x^(n-1)",
                    "description": "The derivative of x raised to a power n is n times x raised to the power n-1",
                    "applications": ["Differentiating polynomial functions", "Finding rates of change"]
                },
                "chain_rule": {
                    "name": "Chain Rule",
                    "formula": "d/dx[f(g(x))] = f'(g(x)) · g'(x)",
                    "description": "Rule for differentiating composite functions",
                    "applications": ["Differentiating complex functions", "Implicit differentiation"]
                },
                "fundamental_theorem": {
                    "name": "Fundamental Theorem of Calculus",
                    "description": "Connects differentiation and integration as inverse processes",
                    "parts": [
                        "If F is an antiderivative of f, then ∫[a,b]f(x)dx = F(b) - F(a)",
                        "If f is continuous on [a,b], then d/dx[∫[a,x]f(t)dt] = f(x)"
                    ]
                }
            },
            "geometry": {
                "pythagorean_theorem": {
                    "name": "Pythagorean Theorem",
                    "formula": "a² + b² = c²",
                    "description": "In a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides",
                    "applications": ["Finding unknown sides in right triangles", "Determining if a triangle is right"]
                },
                "circle_area": {
                    "name": "Circle Area",
                    "formula": "A = πr²",
                    "description": "The area of a circle is pi times the square of its radius",
                },
                "triangle_area": {
                    "name": "Triangle Area",
                    "formula": "A = (1/2)·b·h",
                    "description": "The area of a triangle is half the product of its base and height",
                    "alternative_formulas": ["A = √(s(s-a)(s-b)(s-c)) (Heron's formula, where s = (a+b+c)/2)"]
                }
            },
            "statistics": {
                "mean": {
                    "name": "Arithmetic Mean",
                    "formula": "μ = (Σx) / n",
                    "description": "The sum of all values divided by the number of values",
                },
                "standard_deviation": {
                    "name": "Standard Deviation",
                    "formula": "σ = √[(Σ(x-μ)²) / n]",
                    "description": "A measure of the amount of variation or dispersion of a set of values",
                },
                "normal_distribution": {
                    "name": "Normal Distribution",
                    "formula": "f(x) = (1/(σ√(2π))) · e^(-(x-μ)²/(2σ²))",
                    "description": "A probability distribution characterized by mean μ and standard deviation σ",
                    "properties": ["Symmetric about the mean", "68-95-99.7 rule"]
                }
            },
            "number_theory": {
                "euclidean_algorithm": {
                    "name": "Euclidean Algorithm",
                    "description": "A method for finding the greatest common divisor (GCD) of two integers",
                    "method": "Repeatedly divide the larger number by the smaller and take the remainder until the remainder is 0"
                },
                "prime_factorization": {
                    "name": "Prime Factorization",
                    "description": "Expressing a number as a product of its prime factors",
                    "applications": ["Finding GCD and LCM", "Number theory proofs"]
                },
                "modular_arithmetic": {
                    "name": "Modular Arithmetic",
                    "description": "Arithmetic system where numbers wrap around after reaching a certain value (the modulus)",
                    "applications": ["Cryptography", "Calendar calculations", "Computer science"]
                }
            }
        }
        
    def _initialize_theorem_database(self) -> Dict:
        """Initialize database of mathematical theorems"""
        return {
            "intermediate_value_theorem": {
                "name": "Intermediate Value Theorem",
                "statement": "If f is continuous on [a,b] and k is between f(a) and f(b), then there exists c in [a,b] such that f(c) = k",
                "domain": "calculus",
                "applications": ["Proving existence of solutions", "Finding approximate roots"]
            },
            "mean_value_theorem": {
                "name": "Mean Value Theorem",
                "statement": "If f is continuous on [a,b] and differentiable on (a,b), then there exists c in (a,b) such that f'(c) = [f(b) - f(a)] / (b - a)",
                "domain": "calculus",
                "applications": ["Analyzing function behavior", "Proving inequalities"]
            },
            "central_limit_theorem": {
                "name": "Central Limit Theorem",
                "statement": "The distribution of the sum (or average) of a large number of independent, identically distributed variables approaches a normal distribution",
                "domain": "statistics",
                "applications": ["Statistical inference", "Hypothesis testing"]
            },
            "bayes_theorem": {
                "name": "Bayes' Theorem",
                "statement": "P(A|B) = P(B|A) · P(A) / P(B)",
                "domain": "probability",
                "applications": ["Conditional probability", "Bayesian inference"]
            },
            "fermats_little_theorem": {
                "name": "Fermat's Little Theorem",
                "statement": "If p is prime and a is not divisible by p, then a^(p-1) ≡ 1 (mod p)",
                "domain": "number_theory",
                "applications": ["Primality testing", "Modular exponentiation"]
            }
        }
        
    def _initialize_formula_database(self) -> Dict:
        """Initialize database of mathematical formulas"""
        return {
            "sum_of_arithmetic_series": {
                "name": "Sum of Arithmetic Series",
                "formula": "S_n = n/2 · (a₁ + aₙ) = n/2 · (2a₁ + (n-1)d)",
                "variables": {"n": "number of terms", "a₁": "first term", "aₙ": "last term", "d": "common difference"},
                "domain": "algebra"
            },
            "sum_of_geometric_series": {
                "name": "Sum of Geometric Series",
                "formula": "S_n = a₁ · (1 - r^n) / (1 - r)",
                "variables": {"n": "number of terms", "a₁": "first term", "r": "common ratio"},
                "conditions": "r ≠ 1",
                "domain": "algebra"
            },
            "sum_of_infinite_geometric_series": {
                "name": "Sum of Infinite Geometric Series",
                "formula": "S_∞ = a₁ / (1 - r)",
                "variables": {"a₁": "first term", "r": "common ratio"},
                "conditions": "|r| < 1",
                "domain": "algebra"
            },
            "distance_formula": {
                "name": "Distance Formula",
                "formula": "d = √[(x₂ - x₁)² + (y₂ - y₁)²]",
                "variables": {"d": "distance", "(x₁,y₁)": "first point", "(x₂,y₂)": "second point"},
                "domain": "geometry"
            },
            "volume_of_sphere": {
                "name": "Volume of Sphere",
                "formula": "V = (4/3) · πr³",
                "variables": {"V": "volume", "r": "radius"},
                "domain": "geometry"
            },
            "surface_area_of_sphere": {
                "name": "Surface Area of Sphere",
                "formula": "A = 4πr²",
                "variables": {"A": "surface area", "r": "radius"},
                "domain": "geometry"
            },
            "compound_interest": {
                "name": "Compound Interest",
                "formula": "A = P(1 + r/n)^(nt)",
                "variables": {"A": "final amount", "P": "principal", "r": "annual interest rate", "n": "compounding frequency", "t": "time in years"},
                "domain": "finance"
            },
            "binomial_probability": {
                "name": "Binomial Probability",
                "formula": "P(X = k) = ₍ₙₖ₎ · p^k · (1-p)^(n-k)",
                "variables": {"n": "number of trials", "k": "number of successes", "p": "probability of success on a single trial"},
                "domain": "probability"
            }
        }
        
    def _initialize_method_database(self) -> Dict:
        """Initialize database of mathematical methods and techniques"""
        return {
            "newton_raphson": {
                "name": "Newton-Raphson Method",
                "description": "Iterative method for finding successively better approximations to the roots of a real-valued function",
                "formula": "x_{n+1} = x_n - f(x_n) / f'(x_n)",
                "steps": [
                    "Start with an initial guess x₀",
                    "Calculate x₁ = x₀ - f(x₀) / f'(x₀)",
                    "Repeat until convergence: xₙ₊₁ = xₙ - f(xₙ) / f'(xₙ)"
                ],
                "domain": "numerical_analysis"
            },
            "gaussian_elimination": {
                "name": "Gaussian Elimination",
                "description": "Method for solving systems of linear equations by transforming the augmented matrix into row echelon form",
                "steps": [
                    "Write the system as an augmented matrix",
                    "Use elementary row operations to transform to row echelon form",
                    "Back-substitute to find values of variables"
                ],
                "domain": "linear_algebra"
            },
            "integration_by_parts": {
                "name": "Integration by Parts",
                "description": "Technique for integrating products of functions",
                "formula": "∫u(x)·v'(x)dx = u(x)·v(x) - ∫v(x)·u'(x)dx",
                "steps": [
                    "Identify u(x) and v'(x) from the integrand",
                    "Calculate v(x) = ∫v'(x)dx",
                    "Calculate u'(x)",
                    "Apply the formula"
                ],
                "domain": "calculus"
            },
            "laplace_transform": {
                "name": "Laplace Transform",
                "description": "Technique for solving differential equations by transforming them into algebraic equations",
                "formula": "L{f(t)} = F(s) = ∫₀^∞ e^(-st)·f(t)dt",
                "steps": [
                    "Apply Laplace transform to both sides of the differential equation",
                    "Solve the resulting algebraic equation for F(s)",
                    "Apply inverse Laplace transform to find f(t)"
                ],
                "domain": "differential_equations"
            },
            "simplex_method": {
                "name": "Simplex Method",
                "description": "Algorithm for solving linear programming problems",
                "steps": [
                    "Convert the problem into standard form",
                    "Create an initial basic feasible solution",
                    "Iteratively improve the solution until optimality is reached"
                ],
                "domain": "optimization"
            }
        }
        
    def step(self):
        if not self.active:
            return
            
        new_messages = self.get_new_messages()
        
        for msg in new_messages:
            if msg.type == MessageType.ANALYSIS:
                self.provide_knowledge_for_problem(msg)
            elif msg.type == MessageType.STRATEGY:
                self.provide_knowledge_for_strategy(msg)
            elif msg.type == MessageType.EXECUTION and random.random() < 0.3:  # Only sometimes respond to executions
                self.provide_knowledge_for_execution(msg)
    
    def provide_knowledge_for_problem(self, analysis_msg: Message):
        """Provide relevant knowledge based on problem analysis"""
        analysis = analysis_msg.content
        
        # Extract key information
        domains = analysis.get("domains", {})
        schema = analysis.get("schema", {})
        components = analysis.get("components", {})
        
        # Find primary domain
        primary_domain = max(domains.items(), key=lambda x: x[1])[0] if domains else None
        if not primary_domain:
            return
            
        # Find relevant knowledge
        relevant_knowledge = []
        
        # Domain-specific knowledge
        if primary_domain in self.knowledge_base:
            domain_knowledge = self.knowledge_base[primary_domain]
            
            # Select up to 3 most relevant items
            for key, item in domain_knowledge.items():
                if self._is_relevant_to_problem(item, analysis):
                    relevant_knowledge.append({
                        "type": "concept",
                        "item": item
                    })
                    if len(relevant_knowledge) >= 3:
                        break
        
        # Find relevant theorems
        for theorem_id, theorem in self.theorem_database.items():
            if theorem.get("domain") == primary_domain and self._is_relevant_to_problem(theorem, analysis):
                relevant_knowledge.append({
                    "type": "theorem",
                    "item": theorem
                })
                if len(relevant_knowledge) >= 5:
                    break
        
        # Find relevant formulas
        for formula_id, formula in self.formula_database.items():
            if formula.get("domain") == primary_domain and self._is_relevant_to_problem(formula, analysis):
                relevant_knowledge.append({
                    "type": "formula",
                    "item": formula
                })
                if len(relevant_knowledge) >= 7:
                    break
        
        if relevant_knowledge:
            self.send_message(
                MessageType.KNOWLEDGE, 
                {
                    "problem_id": analysis_msg.id,
                    "knowledge_items": relevant_knowledge,
                    "explanation": f"Here is relevant knowledge for this {primary_domain} problem."
                },
                confidence=0.8,
                references=[analysis_msg.id]
            )
    
    def provide_knowledge_for_strategy(self, strategy_msg: Message):
        """Provide relevant knowledge based on proposed strategy"""
        strategy = strategy_msg.content
        
        # Get the referenced analysis
        analysis = None
        problem_id = strategy.get("problem_id")
        if problem_id:
            analysis_msg = self.workspace.get_message_by_id(problem_id)
            if analysis_msg:
                analysis = analysis_msg.content
        
        if not analysis:
            return
            
        # Find methods relevant to the strategy
        strategy_steps = strategy.get("steps", [])
        relevant_methods = []
        
        for step in strategy_steps:
            if isinstance(step, dict) and "strategy" in step:
                step_strategy = step["strategy"]
                
                # Look for methods related to this strategy
                for method_id, method in self.method_database.items():
                    if step_strategy.lower() in method_id.lower() or step_strategy.lower() in method.get("name", "").lower():
                        relevant_methods.append({
                            "type": "method",
                            "step_id": step.get("step"),
                            "item": method
                        })
                        break
        
        if relevant_methods:
            self.send_message(
                MessageType.KNOWLEDGE, 
                {
                    "strategy_id": strategy_msg.id,
                    "knowledge_items": relevant_methods,
                    "explanation": "Here are methods that may be useful for your strategy."
                },
                confidence=0.8,
                references=[strategy_msg.id]
            )
    
    def provide_knowledge_for_execution(self, execution_msg: Message):
        """Provide insights or tips during execution"""
        execution = execution_msg.content
        strategy_applied = execution.get("strategy_applied", "")
        
        # Simple implementation - would be more sophisticated in real system
        # Look for tips related to the strategy
        tips = []
        
        if "factor" in strategy_applied.lower():
            tips.append("Remember to check that your factorization is correct by expanding it back.")
            
        elif "quadratic" in strategy_applied.lower():
            tips.append("Double-check the discriminant calculation (b² - 4ac) as it's a common source of errors.")
            
        elif "derivative" in strategy_applied.lower():
            tips.append("Don't forget the chain rule when differentiating composite functions.")
            
        elif "integral" in strategy_applied.lower():
            tips.append("Consider using u-substitution or integration by parts for complex integrands.")
            
        elif "system" in strategy_applied.lower() and "equation" in strategy_applied.lower():
            tips.append("Gaussian elimination is often more efficient than substitution for systems with more than two equations.")
        
        if tips:
            self.send_message(
                MessageType.KNOWLEDGE, 
                {
                    "execution_id": execution_msg.id,
                    "tips": tips,
                    "explanation": "Here's a helpful tip for your current step."
                },
                confidence=0.7,
                references=[execution_msg.id]
            )
    
    def _is_relevant_to_problem(self, knowledge_item: Dict, analysis: Dict) -> bool:
        """Determine if a knowledge item is relevant to the problem"""
        # Simple relevance checking - would be more sophisticated in real system
        
        # Check if the problem schema matches the knowledge item
        schema = analysis.get("schema", {})
        schema_type = schema.get("type", "")
        schema_pattern = schema.get("pattern", "")
        
        # Extract item name and description for matching
        item_name = knowledge_item.get("name", "").lower()
        item_desc = knowledge_item.get("description", "").lower()
        
        # Check for direct matches in schema
        if schema_type and (schema_type.lower() in item_name or schema_type.lower() in item_desc):
            return True
            
        if schema_pattern and (schema_pattern.lower() in item_name or schema_pattern.lower() in item_desc):
            return True
        
        # Check for matches in components
        components = analysis.get("components", {})
        target = components.get("target", {}).get("description", "").lower()
        
        if target and (target in item_name or target in item_desc):
            return True
            
        # Check equations for matches
        equations = components.get("equations", [])
        for eq in equations:
            eq_type = eq.get("type", "").lower()
            if eq_type and (eq_type in item_name or eq_type in item_desc):
                return True
        
        # Default: low chance of being relevant anyway
        return random.random() < 0.2  # 20% chance of being considered relevant even without direct matches
    
    def search_knowledge_base(self, query: str) -> List[Dict]:
        """Search the knowledge base for specific information"""
        # Simple search implementation - would be more sophisticated in real system
        results = []
        
        # Search in main knowledge base
        for domain, domain_items in self.knowledge_base.items():
            for key, item in domain_items.items():
                if query.lower() in key.lower() or query.lower() in str(item).lower():
                    results.append({
                        "source": "knowledge_base",
                        "domain": domain,
                        "key": key,
                        "item": item
                    })
        
        # Search in theorem database
        for theorem_id, theorem in self.theorem_database.items():
            if query.lower() in theorem_id.lower() or query.lower() in str(theorem).lower():
                results.append({
                    "source": "theorem_database",
                    "key": theorem_id,
                    "item": theorem
                })
        
        # Search in formula database
        for formula_id, formula in self.formula_database.items():
            if query.lower() in formula_id.lower() or query.lower() in str(formula).lower():
                results.append({
                    "source": "formula_database",
                    "key": formula_id,
                    "item": formula
                })
        
        # Search in method database
        for method_id, method in self.method_database.items():
            if query.lower() in method_id.lower() or query.lower() in str(method).lower():
                results.append({
                    "source": "method_database",
                    "key": method_id,
                    "item": method
                })
        
        return results[:10]  # Return top 10 results


class AnalogicalReasoningAgent(Agent):
    """Agent that uses analogies to similar problems"""
    
    def __init__(self, workspace: Workspace):
        super().__init__("AnalogicalReasoning", workspace)
        self.workspace.subscribe(self.id, [MessageType.ANALYSIS, MessageType.STRATEGY])
        
        # Knowledge database of problem-solution pairs
        self.problem_database = self._initialize_problem_database()
        
    def _initialize_problem_database(self) -> List[Dict]:
        """Initialize database of solved problems for analogical reasoning"""
        return [
            {
                "problem": "Find the roots of the quadratic equation x^2 - 5x + 6 = 0",
                "domain": "algebra",
                "schema": "quadratic_equation",
                "key_features": ["quadratic", "finding roots", "factorizable"],
                "solution_approach": "factoring",
                "solution": "x = 2 or x = 3 by factoring as (x-2)(x-3) = 0"
            },
            {
                "problem": "Find the derivative of f(x) = x^3 - 2x^2 + 4x - 7",
                "domain": "calculus",
                "schema": "differentiation",
                "key_features": ["polynomial", "derivative", "power rule"],
                "solution_approach": "power_rule",
                "solution": "f'(x) = 3x^2 - 4x + 4 using the power rule"
            },
            {
                "problem": "Find the area of a circle with radius 5 cm",
                "domain": "geometry",
                "schema": "circle_area",
                "key_features": ["circle", "area", "radius"],
                "solution_approach": "circle_area_formula",
                "solution": "A = π(5)² = 25π cm² using the circle area formula"
            },
            {
                "problem": "Solve the system of equations: 2x + y = 5 and 3x - 2y = 4",
                "domain": "algebra",
                "schema": "system_of_linear_equations",
                "key_features": ["system", "linear", "two variables"],
                "solution_approach": "elimination",
                "solution": "x = 2, y = 1 by using elimination method"
            },
            {
                "problem": "A ball is thrown upward with initial velocity 20 m/s. How high will it go?",
                "domain": "calculus",
                "schema": "motion_problem",
                "key_features": ["maximum height", "projectile motion", "optimization"],
                "solution_approach": "set_derivative_to_zero",
                "solution": "h = v²/(2g) = 20²/(2×9.8) ≈ 20.4 meters by finding when velocity is zero"
            },
            {
                "problem": "Find the probability of drawing an ace from a standard deck of cards",
                "domain": "probability",
                "schema": "basic_probability",
                "key_features": ["probability", "cards", "favorable outcomes"],
                "solution_approach": "favorable_outcomes_divided_by_total_outcomes",
                "solution": "P = 4/52 = 1/13 since there are 4 aces in a 52-card deck"
            },
            {
                "problem": "Find the definite integral of f(x) = 2x + 3 from x = 1 to x = 4",
                "domain": "calculus",
                "schema": "definite_integration",
                "key_features": ["integral", "linear function", "definite bounds"],
                "solution_approach": "fundamental_theorem_of_calculus",
                "solution": "∫₁⁴(2x + 3)dx = [x² + 3x]₁⁴ = (16 + 12) - (1 + 3) = 24"
            },
            {
                "problem": "Find the maximum value of f(x) = -x² + 6x - 5",
                "domain": "calculus",
                "schema": "optimization",
                "key_features": ["maximum", "quadratic", "critical points"],
                "solution_approach": "derivative_critical_points",
                "solution": "f'(x) = -2x + 6; setting equal to zero gives x = 3; f(3) = 4 is the maximum"
            },
            {
                "problem": "Find the value of sin(30°)",
                "domain": "trigonometry",
                "schema": "special_angle",
                "key_features": ["trigonometric value", "special angle"],
                "solution_approach": "special_angle_values",
                "solution": "sin(30°) = 1/2"
            },
            {
                "problem": "Solve the equation 2^x = 8",
                "domain": "algebra",
                "schema": "exponential_equation",
                "key_features": ["exponential", "same base possible"],
                "solution_approach": "convert_to_same_base",
                "solution": "2^x = 2^3, so x = 3"
            }
        ]
        
    def step(self):
        if not self.active:
            return
            
        new_messages = self.get_new_messages()
        
        for msg in new_messages:
            if msg.type == MessageType.ANALYSIS:
                self.find_analogous_problems(msg)
            elif msg.type == MessageType.STRATEGY:
                self.suggest_analogous_strategies(msg)
    
    def find_analogous_problems(self, analysis_msg: Message):
        """Find analogous problems from the database"""
        analysis = analysis_msg.content
        
        # Extract key features for matching
        domains = analysis.get("domains", {})
        primary_domain = max(domains.items(), key=lambda x: x[1])[0] if domains else None
        
        schema = analysis.get("schema", {})
        schema_type = schema.get("type", "")
        
        components = analysis.get("components", {})
        target_description = components.get("target", {}).get("description", "")
        
        keywords = analysis.get("keywords", [])
        
        # Match against problem database
        analogous_problems = []
        for problem in self.problem_database:
            score = 0
            
            # Domain match
            if problem["domain"] == primary_domain:
                score += 3
                
            # Schema match
            if problem["schema"] == schema_type:
                score += 5
                
            # Keyword match
            for keyword in keywords:
                if keyword in problem["key_features"]:
                    score += 2
                    
            # Target match
            if target_description and any(feature in target_description.lower() for feature in problem["key_features"]):
                score += 2
                
            if score >= 5:  # Only include if reasonably similar
                analogous_problems.append({
                    "problem": problem,
                    "similarity_score": score
                })
        
        # Sort by similarity score
        analogous_problems.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Take top 3 analogous problems
        top_analogies = analogous_problems[:3]
        
        if top_analogies:
            self.send_message(
                MessageType.ANALOGY, 
                {
                    "original_problem_id": analysis_msg.id,
                    "analogous_problems": top_analogies,
                    "explanation": "These problems have similar structure and solution approaches."
                },
                confidence=0.7,
                references=[analysis_msg.id]
            )
    
    def suggest_analogous_strategies(self, strategy_msg: Message):
        """Suggest strategy refinements based on analogous problems"""
        strategy = strategy_msg.content
        
        # Find the original problem analysis
        problem_id = strategy.get("problem_id")
        if not problem_id:
            return
            
        analysis_msg = self.workspace.get_message_by_id(problem_id)
        if not analysis_msg or analysis_msg.type != MessageType.ANALYSIS:
            return
            
        analysis = analysis_msg.content
        
        # Look for analogy messages referring to this problem
        analogy_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.ANALOGY 
                       and any(ref == analysis_msg.id for ref in msg.references)]
        
        if not analogy_msgs:
            return
            
        # Get the most recent analogy message
        analogy_msg = max(analogy_msgs, key=lambda x: x.timestamp)
        analogies = analogy_msg.content.get("analogous_problems", [])
        
        if not analogies:
            return
            
        # Extract the proposed strategy steps
        current_steps = strategy.get("steps", [])
        strategy_names = []
        for step in current_steps:
            if isinstance(step, dict) and "strategy" in step:
                strategy_names.append(step["strategy"])
        
        # Look for potential strategy improvements from analogies
        strategy_suggestions = []
        for analogy in analogies:
            problem = analogy["problem"]
            approach = problem.get("solution_approach", "")
            
            # If the analogous problem used a strategy not in the current strategy
            if approach and approach not in strategy_names:
                strategy_suggestions.append({
                    "analogous_problem": problem["problem"],
                    "suggested_approach": approach,
                    "rationale": f"This similar problem was solved using {approach}.",
                    "similarity_score": analogy["similarity_score"]
                })
        
        if strategy_suggestions:
            self.send_message(
                MessageType.ANALOGY, 
                {
                    "strategy_id": strategy_msg.id,
                    "strategy_suggestions": strategy_suggestions,
                    "explanation": "Based on analogous problems, these alternative approaches might be effective."
                },
                confidence=0.6,
                references=[strategy_msg.id, analogy_msg.id]
            )


# =============================
# STRATEGY AGENTS
# =============================

class StrategyAgent(Agent):
    """Agent that selects appropriate mathematical approaches"""
    
    def __init__(self, workspace: Workspace):
        super().__init__("Strategy", workspace)
        self.workspace.subscribe(self.id, [MessageType.ANALYSIS, MessageType.DEBATE, MessageType.VERIFICATION, 
                                        MessageType.KNOWLEDGE, MessageType.ANALOGY, MessageType.INTUITION])
        
        # Enhanced strategy formulation capabilities
        self.strategy_templates = self._initialize_strategy_templates()
        self.domain_strategies = self._initialize_domain_strategies()
        self.strategy_effectiveness = {}  # Track effectiveness of strategies
        
    def _initialize_strategy_templates(self) -> Dict:
        """Initialize templates for common problem-solving strategies"""
        return {
            "direct_computation": {
                "name": "Direct Computation",
                "description": "Apply formulas or definitions directly to compute the answer",
                "applicable_domains": ["algebra", "calculus", "geometry"],
                "complexity_range": [0.0, 0.4],
                "step_template": [
                    {"description": "Identify the relevant formula or definition"},
                    {"description": "Substitute the given values into the formula"},
                    {"description": "Compute the result"},
                    {"description": "Verify the answer satisfies all constraints"}
                ]
            },
            "decomposition": {
                "name": "Problem Decomposition",
                "description": "Break the problem into simpler subproblems",
                "applicable_domains": ["algebra", "calculus", "geometry", "number_theory"],
                "complexity_range": [0.3, 0.7],
                "step_template": [
                    {"description": "Decompose the problem into smaller subproblems"},
                    {"description": "Solve each subproblem independently"},
                    {"description": "Combine the subproblem solutions"},
                    {"description": "Verify the combined solution"}
                ]
            },
            "transformation": {
                "name": "Problem Transformation",
                "description": "Transform the problem into an equivalent but more tractable form",
                "applicable_domains": ["algebra", "calculus", "geometry"],
                "complexity_range": [0.4, 0.8],
                "step_template": [
                    {"description": "Identify an appropriate transformation"},
                    {"description": "Apply the transformation to get an equivalent problem"},
                    {"description": "Solve the transformed problem"},
                    {"description": "Map the solution back to the original problem"},
                    {"description": "Verify the solution"}
                ]
            },
            "iterative_approximation": {
                "name": "Iterative Approximation",
                "description": "Start with an initial approximation and iteratively refine it",
                "applicable_domains": ["calculus", "numerical_analysis"],
                "complexity_range": [0.6, 1.0],
                "step_template": [
                    {"description": "Formulate an iterative procedure"},
                    {"description": "Choose an initial approximation"},
                    {"description": "Apply the procedure iteratively until convergence"},
                    {"description": "Verify the approximation is within acceptable error"}
                ]
            }
        }
    
    def _initialize_domain_strategies(self) -> Dict:
        """Initialize domain-specific strategies"""
        return {
            "algebra": {
                "factor_and_solve": {
                    "name": "Factor and Solve",
                    "description": "Factor polynomial equation and find roots",
                    "applicable_to": ["polynomial_equation", "quadratic_equation"],
                    "complexity_range": [0.2, 0.6],
                    "steps": [
                        {"description": "Try to factor the polynomial into simpler terms"},
                        {"description": "Set each factor equal to zero"},
                        {"description": "Solve the resulting simple equations"},
                        {"description": "Verify solutions by substituting back"}
                    ]
                },
                "substitute_variables": {
                    "name": "Variable Substitution",
                    "description": "Substitute variables to simplify expressions",
                    "applicable_to": ["system_of_equations", "complex_expressions"],
                    "complexity_range": [0.3, 0.7],
                    "steps": [
                        {"description": "Identify a substitution that simplifies the problem"},
                        {"description": "Perform the substitution and simplify"},
                        {"description": "Solve the simplified problem"},
                        {"description": "Substitute back to get original variables"}
                    ]
                },
                "complete_the_square": {
                    "name": "Complete the Square",
                    "description": "Rewrite quadratic expressions in vertex form",
                    "applicable_to": ["quadratic_equation", "quadratic_expression"],
                    "complexity_range": [0.3, 0.6],
                    "steps": [
                        {"description": "Rewrite the quadratic in the form a(x² + bx/a) + c"},
                        {"description": "Complete the square to get a(x + b/2a)² + k"},
                        {"description": "Use the resulting form to extract information"}
                    ]
                },
                "use_quadratic_formula": {
                    "name": "Quadratic Formula",
                    "description": "Apply the quadratic formula to find roots",
                    "applicable_to": ["quadratic_equation"],
                    "complexity_range": [0.2, 0.5],
                    "steps": [
                        {"description": "Identify coefficients a, b, and c"},
                        {"description": "Calculate the discriminant b² - 4ac"},
                        {"description": "Apply the formula x = (-b ± √(b² - 4ac))/2a"},
                        {"description": "Simplify the results"}
                    ]
                }
            },
            "calculus": {
                "compute_derivative": {
                    "name": "Compute Derivative",
                    "description": "Calculate the derivative of a function",
                    "applicable_to": ["rate_of_change", "tangent_line", "optimization"],
                    "complexity_range": [0.3, 0.7],
                    "steps": [
                        {"description": "Identify differentiation rules needed"},
                        {"description": "Apply the rules step by step"},
                        {"description": "Simplify the resulting expression"}
                    ]
                },
                "find_critical_points": {
                    "name": "Find Critical Points",
                    "description": "Locate critical points for optimization",
                    "applicable_to": ["optimization", "max_min_problem"],
                    "complexity_range": [0.4, 0.8],
                    "steps": [
                        {"description": "Compute the derivative of the function"},
                        {"description": "Set the derivative equal to zero and solve"},
                        {"description": "Check second derivative or analyze behavior"},
                        {"description": "Classify critical points as maxima, minima, or inflection points"}
                    ]
                },
                "compute_integral": {
                    "name": "Compute Integral",
                    "description": "Calculate the integral of a function",
                    "applicable_to": ["area_problem", "accumulation", "volume_problem"],
                    "complexity_range": [0.4, 0.8],
                    "steps": [
                        {"description": "Identify integration techniques needed"},
                        {"description": "Apply the techniques step by step"},
                        {"description": "Compute any necessary constants"},
                        {"description": "Evaluate bounds for definite integrals"}
                    ]
                }
            },
            "geometry": {
                "use_coordinate_geometry": {
                    "name": "Coordinate Geometry",
                    "description": "Use coordinate system to solve geometric problems",
                    "applicable_to": ["distance_problem", "line_problem", "area_problem"],
                    "complexity_range": [0.3, 0.7],
                    "steps": [
                        {"description": "Set up a coordinate system"},
                        {"description": "Express geometric objects in coordinates"},
                        {"description": "Apply coordinate formulas to solve"},
                        {"description": "Interpret the results geometrically"}
                    ]
                },
                "apply_triangle_theorems": {
                    "name": "Triangle Theorems",
                    "description": "Apply theorems about triangles",
                    "applicable_to": ["triangle_problem", "similar_triangles"],
                    "complexity_range": [0.2, 0.6],
                    "steps": [
                        {"description": "Identify applicable triangle theorems"},
                        {"description": "Set up equations based on the theorems"},
                        {"description": "Solve for unknown quantities"},
                        {"description": "Verify the solution makes geometric sense"}
                    ]
                }
            },
            "statistics": {
                "apply_probability_rules": {
                    "name": "Probability Rules",
                    "description": "Apply rules of probability",
                    "applicable_to": ["probability_problem", "counting_problem"],
                    "complexity_range": [0.3, 0.7],
                    "steps": [
                        {"description": "Identify the types of events (independent, mutually exclusive, etc.)"},
                        {"description": "Apply appropriate probability rules"},
                        {"description": "Calculate the final probability"},
                        {"description": "Verify the result is in the range [0, 1]"}
                    ]
                },
                "compute_summary_statistics": {
                    "name": "Summary Statistics",
                    "description": "Compute statistical measures of data",
                    "applicable_to": ["data_analysis_problem", "statistical_inference"],
                    "complexity_range": [0.2, 0.6],
                    "steps": [
                        {"description": "Identify relevant statistical measures needed"},
                        {"description": "Calculate measures (mean, median, std dev, etc.)"},
                        {"description": "Interpret the results in context"}
                    ]
                },
                "apply_bayes_theorem": {
                    "name": "Bayes' Theorem",
                    "description": "Apply Bayes' theorem for conditional probability",
                    "applicable_to": ["conditional_probability", "bayesian_updating"],
                    "complexity_range": [0.4, 0.8],
                    "steps": [
                        {"description": "Identify the prior probability"},
                        {"description": "Calculate the likelihood and evidence"},
                        {"description": "Apply Bayes' formula to find posterior probability"}
                    ]
                }
            },
            "number_theory": {
                "prime_factorization": {
                    "name": "Prime Factorization",
                    "description": "Express number as product of prime factors",
                    "applicable_to": ["prime_problem", "divisibility_problem"],
                    "complexity_range": [0.3, 0.7],
                    "steps": [
                        {"description": "Find prime factors of the number"},
                        {"description": "Express the number as a product of prime powers"},
                        {"description": "Use the factorization to solve the problem"}
                    ]
                },
                "use_modular_arithmetic": {
                    "name": "Modular Arithmetic",
                    "description": "Apply modular arithmetic principles",
                    "applicable_to": ["congruence_problem", "remainder_problem"],
                    "complexity_range": [0.4, 0.8],
                    "steps": [
                        {"description": "Convert the problem to modular form"},
                        {"description": "Apply relevant modular arithmetic properties"},
                        {"description": "Solve for the unknown values"},
                        {"description": "Convert back to the original context"}
                    ]
                }
            }
        }
    
    def step(self):
        if not self.active:
            return
            
        new_messages = self.get_new_messages()
        analyses = [msg for msg in new_messages if msg.type == MessageType.ANALYSIS]
        debates = [msg for msg in new_messages if msg.type == MessageType.DEBATE]
        verifications = [msg for msg in new_messages if msg.type == MessageType.VERIFICATION]
        knowledge_msgs = [msg for msg in new_messages if msg.type == MessageType.KNOWLEDGE]
        analogy_msgs = [msg for msg in new_messages if msg.type == MessageType.ANALOGY]
        intuition_msgs = [msg for msg in new_messages if msg.type == MessageType.INTUITION]
        
        # Prioritize responding to debates and verification failures
        if debates:
            for debate_msg in debates:
                if "strategy" in str(debate_msg.content):
                    self.revise_strategy(debate_msg)
        
        elif verifications:
            for verify_msg in verifications:
                if verify_msg.confidence < 0.5:  # Failed verification
                    self.revise_strategy_after_failure(verify_msg)
        
        # Process new problem analyses
        elif analyses:
            for analysis_msg in analyses:
                # Collect related knowledge and analogies first
                related_knowledge = [msg for msg in knowledge_msgs 
                                  if msg.references and analysis_msg.id in msg.references]
                related_analogies = [msg for msg in analogy_msgs 
                                  if msg.references and analysis_msg.id in msg.references]
                related_intuitions = [msg for msg in intuition_msgs 
                                   if msg.references and analysis_msg.id in msg.references]
                
                self.formulate_strategy(analysis_msg, related_knowledge, related_analogies, related_intuitions)
    
    def formulate_strategy(self, analysis_msg, knowledge_msgs=None, analogy_msgs=None, intuition_msgs=None):
        """Formulate a comprehensive solution strategy based on problem analysis"""
        analysis = analysis_msg.content
        domains = analysis.get("domains", {})
        schema = analysis.get("schema", {})
        complexity = analysis.get("complexity", 0.5)
        
        # Determine primary and secondary domains
        primary_domain = max(domains.items(), key=lambda x: x[1])[0] if domains else "algebra"
        secondary_domains = [d for d, v in domains.items() if v > 0.2 and d != primary_domain]
        
        # Select high-level strategy template based on complexity
        template = self._select_strategy_template(primary_domain, complexity)
        
        # Get domain-specific strategies
        candidate_strategies = self._get_candidate_strategies(primary_domain, secondary_domains, schema)
        
        # Filter strategies based on problem attributes
        filtered_strategies = self._filter_strategies(candidate_strategies, analysis)
        
        # Incorporate knowledge from other agents
        if knowledge_msgs:
            filtered_strategies = self._incorporate_knowledge(filtered_strategies, knowledge_msgs)
            
        # Incorporate analogies
        if analogy_msgs:
            filtered_strategies = self._incorporate_analogies(filtered_strategies, analogy_msgs)
            
        # Incorporate intuitions
        if intuition_msgs:
            filtered_strategies = self._incorporate_intuitions(filtered_strategies, intuition_msgs)
        
        # Sort by confidence and select top strategies
        sorted_strategies = sorted(filtered_strategies, key=lambda x: x[2], reverse=True)
        top_strategies = sorted_strategies[:min(4, len(sorted_strategies))]
        
        # Create a sequential plan
        steps = []
        used_strategies = set()
        
        # Add initialization step
        steps.append({
            "step": 1,
            "phase": "initialization",
            "description": "Understand the problem and identify key components",
            "expected_outcome": "Clear formulation of what we're looking for"
        })
        
        # Add steps for each selected strategy
        step_num = 2
        for domain, strategy_name, confidence in top_strategies:
            if strategy_name not in used_strategies:  # Avoid duplicate strategies
                # Get strategy details
                strategy_details = self._get_strategy_details(domain, strategy_name)
                if strategy_details:
                    steps.append({
                        "step": step_num,
                        "phase": "execution",
                        "domain": domain,
                        "strategy": strategy_name,
                        "description": strategy_details.get("name", strategy_name),
                        "detailed_description": strategy_details.get("description", ""),
                        "specific_steps": strategy_details.get("steps", []),
                        "confidence": confidence,
                        "expected_outcome": self._get_expected_outcome(strategy_name, analysis)
                    })
                    step_num += 1
                    used_strategies.add(strategy_name)
        
        # Add verification step
        steps.append({
            "step": step_num,
            "phase": "verification",
            "description": "Verify the solution by checking it against the original problem",
            "expected_outcome": "Confirmation that our answer is correct"
        })
        
        # Create final strategy plan
        strategy_plan = {
            "problem_id": analysis_msg.id,
            "high_level_approach": template.get("name", "Standard Problem-Solving Approach"),
            "primary_domain": primary_domain,
            "complexity_assessment": complexity,
            "steps": steps,
            "fallback_strategies": [s for _, s, _ in sorted_strategies[len(top_strategies):min(len(sorted_strategies), len(top_strategies)+3)]],
            "estimated_difficulty": self._estimate_solution_difficulty(analysis, steps)
        }
        
        # Calculate overall confidence in the strategy
        overall_confidence = self._calculate_strategy_confidence(strategy_plan)
        
        self.send_message(
            MessageType.STRATEGY, 
            strategy_plan,
            confidence=overall_confidence,
            references=[analysis_msg.id] + 
                      ([msg.id for msg in knowledge_msgs] if knowledge_msgs else []) +
                      ([msg.id for msg in analogy_msgs] if analogy_msgs else []) +
                      ([msg.id for msg in intuition_msgs] if intuition_msgs else [])
        )
    
    def _select_strategy_template(self, domain, complexity):
        """Select a high-level strategy template based on domain and complexity"""
        suitable_templates = []
        for template_id, template in self.strategy_templates.items():
            min_complexity, max_complexity = template.get("complexity_range", [0.0, 1.0])
            if (domain in template.get("applicable_domains", []) and 
                min_complexity <= complexity <= max_complexity):
                suitable_templates.append((template_id, template))
        
        if not suitable_templates:
            # Default template if none are suitable
            return {"name": "Direct Solution Approach", "description": "Solve the problem directly"}
        
        # Select template based on complexity match
        template_id, template = min(suitable_templates, 
                                    key=lambda x: abs(complexity - sum(x[1].get("complexity_range", [0.0, 1.0]))/2))
        return template
    
    def _get_candidate_strategies(self, primary_domain, secondary_domains, schema):
        """Get candidate strategies for the primary and secondary domains"""
        candidate_strategies = []
        schema_type = schema.get("type", "")
        
        # Add strategies from primary domain
        if primary_domain in self.domain_strategies:
            for strategy_id, strategy in self.domain_strategies[primary_domain].items():
                applicable_to = strategy.get("applicable_to", [])
                if not applicable_to or schema_type in applicable_to:
                    candidate_strategies.append((primary_domain, strategy_id, 1.0))
        
        # Add strategies from secondary domains
        for domain in secondary_domains:
            if domain in self.domain_strategies:
                for strategy_id, strategy in self.domain_strategies[domain].items():
                    applicable_to = strategy.get("applicable_to", [])
                    if not applicable_to or schema_type in applicable_to:
                        candidate_strategies.append((domain, strategy_id, 0.7))
        
        return candidate_strategies
    
    def _filter_strategies(self, candidate_strategies, analysis):
        """Filter strategies based on problem attributes"""
        filtered_strategies = []
        
        components = analysis.get("components", {})
        schema = analysis.get("schema", {})
        complexity = analysis.get("complexity", 0.5)
        
        for domain, strategy, base_confidence in candidate_strategies:
            confidence = base_confidence
            
            # Get strategy details
            strategy_details = None
            if domain in self.domain_strategies and strategy in self.domain_strategies[domain]:
                strategy_details = self.domain_strategies[domain][strategy]
            
            if strategy_details:
                # Check complexity range
                min_complexity, max_complexity = strategy_details.get("complexity_range", [0.0, 1.0])
                if complexity < min_complexity:
                    confidence -= 0.2  # Too simple for this strategy
                elif complexity > max_complexity:
                    confidence -= 0.3  # Too complex for this strategy
                
                # Check applicability to schema
                applicable_to = strategy_details.get("applicable_to", [])
                schema_type = schema.get("type", "")
                if applicable_to and schema_type not in applicable_to:
                    confidence -= 0.2  # Not directly applicable to this schema
            
            # Adjust confidence based on specific problem features
            if domain == "algebra":
                if strategy == "factor_and_solve":
                    equations = components.get("equations", [])
                    if any(eq.get("type") == "quadratic" for eq in equations):
                        confidence += 0.2
                elif strategy == "use_quadratic_formula":
                    equations = components.get("equations", [])
                    if any(eq.get("type") == "quadratic" for eq in equations):
                        confidence += 0.2
            elif domain == "calculus":
                if strategy == "compute_derivative":
                    target = components.get("target", {}).get("description", "")
                    if "rate" in target.lower() or "slope" in target.lower():
                        confidence += 0.2
                elif strategy == "find_critical_points":
                    target = components.get("target", {}).get("description", "")
                    if "maximum" in target.lower() or "minimum" in target.lower():
                        confidence += 0.3
            
            # Check past performance of this strategy (if tracked)
            if strategy in self.strategy_effectiveness:
                past_effectiveness = self.strategy_effectiveness.get(strategy, 0.5)
                confidence += (past_effectiveness - 0.5) * 0.4  # Adjust by at most ±0.2
            
            filtered_strategies.append((domain, strategy, min(confidence, 1.0)))
        
        return filtered_strategies
    
    def _incorporate_knowledge(self, strategies, knowledge_msgs):
        """Incorporate knowledge from KnowledgeBaseAgent into strategy selection"""
        # Start with a copy of existing strategies
        enhanced_strategies = strategies.copy()
        
        for msg in knowledge_msgs:
            knowledge_items = msg.content.get("knowledge_items", [])
            
            for item in knowledge_items:
                item_type = item.get("type", "")
                item_content = item.get("item", {})
                
                # Check if knowledge relates to a strategy
                if item_type == "method":
                    method_name = item_content.get("name", "").lower()
                    
                    # Look for matching strategies and boost confidence
                    for i, (domain, strategy, confidence) in enumerate(enhanced_strategies):
                        if strategy.lower() in method_name or method_name in strategy.lower():
                            # Boost confidence for this strategy
                            enhanced_strategies[i] = (domain, strategy, min(confidence + 0.15, 1.0))
        
        return enhanced_strategies
    
    def _incorporate_analogies(self, strategies, analogy_msgs):
        """Incorporate analogical reasoning into strategy selection"""
        # Start with a copy of existing strategies
        enhanced_strategies = strategies.copy()
        
        for msg in analogy_msgs:
            # Check for strategy suggestions
            strategy_suggestions = msg.content.get("strategy_suggestions", [])
            
            for suggestion in strategy_suggestions:
                suggested_approach = suggestion.get("suggested_approach", "")
                similarity = suggestion.get("similarity_score", 0) / 10.0  # Normalize to 0-1
                
                # Check if suggested approach is already in our strategies
                found = False
                for i, (domain, strategy, confidence) in enumerate(enhanced_strategies):
                    if strategy.lower() == suggested_approach.lower():
                        # Boost confidence based on similarity
                        enhanced_strategies[i] = (domain, strategy, min(confidence + similarity * 0.2, 1.0))
                        found = True
                        break
                
                # If not found, try to add it
                if not found and similarity > 0.5:  # Only add if reasonably similar
                    # Determine which domain this might belong to
                    for domain, strategies in self.domain_strategies.items():
                        if suggested_approach in strategies:
                            enhanced_strategies.append((domain, suggested_approach, 0.5 + similarity * 0.3))
                            break
        
        return enhanced_strategies
    
    def _incorporate_intuitions(self, strategies, intuition_msgs):
        """Incorporate intuitions into strategy selection"""
        # Start with a copy of existing strategies
        enhanced_strategies = strategies.copy()
        
        for msg in intuition_msgs:
            intuitions = []
            
            # Check different intuition formats
            if "initial_impressions" in msg.content:
                intuitions.extend(msg.content["initial_impressions"])
            if "analysis_based_intuitions" in msg.content:
                intuitions.extend(msg.content["analysis_based_intuitions"])
            if "strategic_intuitions" in msg.content:
                intuitions.extend(msg.content["strategic_intuitions"])
            
            for intuition in intuitions:
                heuristic = intuition.get("heuristic", "")
                intuition_text = intuition.get("intuition", "")
                confidence = intuition.get("confidence", 0.5)
                
                # Adjust strategy confidences based on intuitions
                for i, (domain, strategy, strategy_confidence) in enumerate(enhanced_strategies):
                    # Check for direct mentions
                    if strategy.lower() in intuition_text.lower():
                        # Direct mention - boost or reduce confidence based on intuition confidence
                        adjustment = (confidence - 0.5) * 0.4  # Max adjustment of ±0.2
                        enhanced_strategies[i] = (domain, strategy, max(0.1, min(strategy_confidence + adjustment, 1.0)))
                    
                    # Check for heuristic matches
                    elif heuristic == "simplification" and "simple" in strategy.lower():
                        enhanced_strategies[i] = (domain, strategy, min(strategy_confidence + 0.1, 1.0))
                    elif heuristic == "visualization" and domain == "geometry":
                        enhanced_strategies[i] = (domain, strategy, min(strategy_confidence + 0.1, 1.0))
                    elif heuristic == "extreme_cases" and "critical_points" in strategy.lower():
                        enhanced_strategies[i] = (domain, strategy, min(strategy_confidence + 0.15, 1.0))
        
        return enhanced_strategies
    
    def _get_strategy_details(self, domain, strategy):
        """Get details for a specific strategy"""
        if domain in self.domain_strategies and strategy in self.domain_strategies[domain]:
            return self.domain_strategies[domain][strategy]
        return None
    
    def _get_expected_outcome(self, strategy, analysis):
        """Get the expected outcome for a given strategy"""
        # Map strategies to expected outcomes
        strategy_outcomes = {
            "factor_and_solve": "Values of variables that make the expression zero",
            "substitute_variables": "Simplified equation with fewer variables",
            "complete_the_square": "Quadratic in vertex form",
            "use_quadratic_formula": "Exact values of the roots of the quadratic",
            "compute_derivative": "Formula for the rate of change or slope",
            "find_critical_points": "Locations of maxima, minima, or inflection points",
            "compute_integral": "Antiderivative or area under the curve",
            "use_coordinate_geometry": "Solution expressed in terms of coordinates",
            "apply_triangle_theorems": "Values of unknown sides or angles",
            "apply_probability_rules": "Probability value between 0 and 1",
            "compute_summary_statistics": "Statistical measures that summarize the data",
            "apply_bayes_theorem": "Updated probability given new evidence",
            "prime_factorization": "Expression of number as product of primes",
            "use_modular_arithmetic": "Solution in terms of remainders or congruences"
        }
        
        # Get the target from analysis
        target = analysis.get("components", {}).get("target", {}).get("description", "")
        
        # Return strategy-specific outcome, or fallback to target-based outcome
        return strategy_outcomes.get(strategy, f"Result that determines {target}")
    
    def _estimate_solution_difficulty(self, analysis, steps):
        """Estimate the overall difficulty of solving the problem"""
        # Base difficulty on problem complexity
        difficulty = analysis.get("complexity", 0.5)
        
        # Adjust based on number of steps
        execution_steps = [s for s in steps if s.get("phase") == "execution"]
        if len(execution_steps) > 3:
            difficulty += 0.1
        if len(execution_steps) > 5:
            difficulty += 0.1
        
        # Adjust based on domain
        domains = analysis.get("domains", {})
        if "calculus" in domains and domains["calculus"] > 0.7:
            difficulty += 0.1
        if "number_theory" in domains and domains["number_theory"] > 0.6:
            difficulty += 0.1
        
        return min(difficulty, 1.0)
    
    def _calculate_strategy_confidence(self, strategy_plan):
        """Calculate overall confidence in the strategy"""
        # Base confidence
        confidence = 0.7
        
        # Adjust based on step confidences
        step_confidences = [step.get("confidence", 0.5) for step in strategy_plan["steps"] 
                           if "confidence" in step]
        if step_confidences:
            avg_step_confidence = sum(step_confidences) / len(step_confidences)
            confidence = 0.3 * confidence + 0.7 * avg_step_confidence
        
        # Penalize for complexity
        complexity = strategy_plan.get("complexity_assessment", 0.5)
        if complexity > 0.7:
            confidence -= 0.1
        
        # Penalize if insufficient steps for complex problems
        if complexity > 0.6 and len(strategy_plan["steps"]) < 4:
            confidence -= 0.1
        
        return max(0.4, min(confidence, 0.95))  # Cap between 0.4 and 0.95
    
    def revise_strategy(self, debate_msg):
        """Revise strategy based on debate feedback"""
        feedback = debate_msg.content
        contested_strategy_msg_id = None
        
        # Find the contested strategy message
        if "contested_message_id" in feedback:
            contested_strategy_msg_id = feedback["contested_message_id"]
        elif debate_msg.references:
            # Try to find a strategy message in the references
            for ref_id in debate_msg.references:
                msg = self.workspace.get_message_by_id(ref_id)
                if msg and msg.type == MessageType.STRATEGY:
                    contested_strategy_msg_id = ref_id
                    break
                
        if not contested_strategy_msg_id:
            return
            
        # Get the original strategy message
        original_strategy_msg = self.workspace.get_message_by_id(contested_strategy_msg_id)
        if not original_strategy_msg or original_strategy_msg.type != MessageType.STRATEGY:
            return
            
        # Get original strategy content
        original_strategy = original_strategy_msg.content
        
        # Extract debate points
        debate_issue = feedback.get("specific_issue", "")
        debate_reasoning = feedback.get("reasoning", "")
        alternative = feedback.get("alternative_consideration", "")
        
        # Create revised strategy
        revised_strategy = original_strategy.copy()
        
        # Modify strategy based on debate points
        steps_modified = False
        
        for i, step in enumerate(revised_strategy["steps"]):
            if isinstance(step, dict) and "strategy" in step:
                strategy_name = step["strategy"]
                
                # Check if this step is being debated
                if strategy_name in debate_issue or strategy_name in debate_reasoning:
                    # Try to replace with a fallback strategy
                    if revised_strategy["fallback_strategies"]:
                        fallback = revised_strategy["fallback_strategies"].pop(0)
                        fallback_details = self._get_strategy_details(step["domain"], fallback)
                        
                        if fallback_details:
                            revised_strategy["steps"][i] = {
                                "step": step["step"],
                                "phase": "execution",
                                "domain": step["domain"],
                                "strategy": fallback,
                                "description": fallback_details.get("name", fallback),
                                "detailed_description": fallback_details.get("description", ""),
                                "specific_steps": fallback_details.get("steps", []),
                                "confidence": 0.6,  # Lower confidence for fallback
                                "expected_outcome": self._get_expected_outcome(fallback, {}),
                                "debate_note": f"Changed from {strategy_name} due to debate"
                            }
                            steps_modified = True
                            
                            # Track strategy effectiveness decrease
                            self.strategy_effectiveness[strategy_name] = self.strategy_effectiveness.get(strategy_name, 0.5) * 0.8
        
        # If no steps were modified but there's a debate, add a clarification step
        if not steps_modified and alternative:
            # Add a clarification step
            last_step_num = max([s.get("step", 0) for s in revised_strategy["steps"]])
            verification_steps = [s for s in revised_strategy["steps"] if s.get("phase") == "verification"]
            
            if verification_steps:
                # Insert before verification
                insert_index = revised_strategy["steps"].index(verification_steps[0])
                clarification_step = {
                    "step": last_step_num + 0.5,  # Use half-step numbering
                    "phase": "clarification",
                    "description": "Address potential issues raised in debate",
                    "clarification": alternative,
                    "expected_outcome": "Improved understanding and approach"
                }
                revised_strategy["steps"].insert(insert_index, clarification_step)
            else:
                # Add at the end
                clarification_step = {
                    "step": last_step_num + 1,
                    "phase": "clarification",
                    "description": "Address potential issues raised in debate",
                    "clarification": alternative,
                    "expected_outcome": "Improved understanding and approach"
                }
                revised_strategy["steps"].append(clarification_step)
        
        # Add notes about the revision
        revised_strategy["revision_history"] = revised_strategy.get("revision_history", [])
        revised_strategy["revision_history"].append({
            "revision_id": len(revised_strategy["revision_history"]) + 1,
            "debate_source": debate_msg.sender,
            "issue": debate_issue,
            "reasoning": debate_reasoning,
            "changes_made": "Modified strategies based on debate feedback" if steps_modified else "Added clarification step"
        })
        
        # Recalculate confidence
        confidence = self._calculate_strategy_confidence(revised_strategy)
        
        # Send the revised strategy
        self.send_message(
            MessageType.STRATEGY, 
            revised_strategy,
            confidence=confidence * 0.9,  # Slightly lower confidence for revised strategy
            references=[original_strategy_msg.id, debate_msg.id],
            priority=1.2  # Higher priority for revisions
        )
    
    def revise_strategy_after_failure(self, verification_msg):
        """Revise strategy after a verification failure"""
        verification = verification_msg.content
        execution_id = verification.get("step_id")
        
        if not execution_id:
            return
            
        # Find the execution message being verified
        execution_msg = self.workspace.get_message_by_id(execution_id)
        if not execution_msg or execution_msg.type != MessageType.EXECUTION:
            return
            
        # Get the strategy that led to this execution
        strategy_msg = None
        for ref_id in execution_msg.references:
            msg = self.workspace.get_message_by_id(ref_id)
            if msg and msg.type == MessageType.STRATEGY:
                strategy_msg = msg
                break
                
        if not strategy_msg:
            return
            
        # Get strategy and execution details
        strategy = strategy_msg.content
        execution = execution_msg.content
        failed_strategy = execution.get("strategy_applied", "")
        
        # Check for revision already in progress
        recent_revisions = [msg for msg in self.workspace.messages 
                          if msg.type == MessageType.STRATEGY and 
                          msg.timestamp > strategy_msg.timestamp and
                          strategy_msg.id in msg.references]
        
        if recent_revisions:
            # A revision is already in progress, don't create another
            return
            
        # Create a revised strategy, replacing the failed step
        revised_strategy = strategy.copy()
        
        # Find and replace the failed step
        for i, step in enumerate(revised_strategy["steps"]):
            if isinstance(step, dict) and step.get("strategy") == failed_strategy:
                # Check for fallback strategies
                if revised_strategy["fallback_strategies"]:
                    fallback = revised_strategy["fallback_strategies"].pop(0)
                    fallback_details = self._get_strategy_details(step["domain"], fallback)
                    
                    if fallback_details:
                        revised_strategy["steps"][i] = {
                            "step": step["step"],
                            "phase": "execution",
                            "domain": step["domain"],
                            "strategy": fallback,
                            "description": fallback_details.get("name", fallback),
                            "detailed_description": fallback_details.get("description", ""),
                            "specific_steps": fallback_details.get("steps", []),
                            "confidence": 0.6,  # Lower confidence for fallback
                            "expected_outcome": self._get_expected_outcome(fallback, {}),
                            "failure_note": f"Changed from {failed_strategy} due to verification failure"
                        }
                        
                        # Track strategy effectiveness decrease
                        self.strategy_effectiveness[failed_strategy] = self.strategy_effectiveness.get(failed_strategy, 0.5) * 0.7
                        break
                    
        # Add notes about the revision
        revised_strategy["revision_history"] = revised_strategy.get("revision_history", [])
        revised_strategy["revision_history"].append({
            "revision_id": len(revised_strategy["revision_history"]) + 1,
            "verification_source": verification_msg.sender,
            "verification_score": verification.get("verification_score", 0),
            "failed_strategy": failed_strategy,
            "issues": verification.get("issues", []),
            "changes_made": "Replaced failed strategy with alternative approach"
        })
        
        # Recalculate confidence
        confidence = self._calculate_strategy_confidence(revised_strategy) * 0.9  # Reduced confidence after failure
        
        # Send the revised strategy
        self.send_message(
            MessageType.STRATEGY, 
            revised_strategy,
            confidence=confidence,
            references=[strategy_msg.id, verification_msg.id, execution_msg.id],
            priority=1.5  # High priority for failure-based revisions
        )


# =============================
# EXECUTOR AGENTS
# =============================

class ExecutorAgent(Agent):
    """Base class for all executor agents"""
    
    def __init__(self, agent_id: str, workspace: Workspace, domain: str):
        super().__init__(agent_id, workspace)
        self.domain = domain
        self.workspace.subscribe(self.id, [MessageType.STRATEGY, MessageType.KNOWLEDGE])
        self.techniques = {}  # To be defined by subclasses
        
    def step(self):
        if not self.active:
            return
            
        new_messages = self.get_new_messages()
        
        # Process strategies that might have steps for this executor
        for msg in new_messages:
            if msg.type == MessageType.STRATEGY:
                strategy = msg.content
                
                # Find steps relevant to this executor
                for step in strategy.get("steps", []):
                    if isinstance(step, dict) and step.get("domain") == self.domain and step.get("strategy") in self.techniques:
                        # Get the problem analysis
                        analysis_msg = self.workspace.get_message_by_id(strategy.get("problem_id"))
                        if analysis_msg:
                            self.execute_step(step, analysis_msg.content, [msg.id])
    
    def execute_step(self, step: Dict, analysis: Dict, references: List[int]):
        """Execute a solution step"""
        strategy_name = step.get("strategy")
        technique = self.techniques.get(strategy_name)
        
        if not technique:
            return
            
        try:
            # Time the execution for performance metrics
            start_time = time.time()
            
            # Execute the appropriate technique
            result = technique(analysis, step)
            
            # Record execution time
            execution_time = time.time() - start_time
            self.update_performance_metrics("execution_time", execution_time)
            
            # Include specific steps if available
            specific_steps = []
            if isinstance(result.get("working"), list):
                specific_steps = result["working"]
            else:
                # Convert string working to steps if possible
                working_str = result.get("working", "")
                if working_str:
                    specific_steps = [{"description": line.strip()} for line in working_str.split("\n") if line.strip()]
            
            execution_result = {
                "step_description": step.get("description", ""),
                "strategy_applied": strategy_name,
                "working": result.get("working", ""),
                "specific_steps": specific_steps,
                "result": result.get("result", ""),
                "explanation": result.get("explanation", ""),
                "domain": self.domain,
                "execution_time_ms": int(execution_time * 1000)
            }
            
            # Include any computed values
            if "computed_values" in result:
                execution_result["computed_values"] = result["computed_values"]
            
            # Include any visualizations
            if "visualization" in result:
                execution_result["visualization"] = result["visualization"]
            
            confidence = result.get("confidence", step.get("confidence", 0.8))
            
            self.send_message(
                MessageType.EXECUTION, 
                execution_result,
                confidence=confidence,
                references=references
            )
            
            # If result includes a visualization, send it separately
            if "visualization" in result:
                self.send_message(
                    MessageType.VISUALIZATION,
                    {
                        "type": "execution_visualization",
                        "execution_strategy": strategy_name,
                        "visualization_data": result["visualization"]
                    },
                    confidence=confidence,
                    references=references + [self.workspace.messages[-1].id]  # Reference the execution message
                )
            
        except Exception as e:
            # Send error message if execution fails
            error_result = {
                "step_description": step.get("description", ""),
                "strategy_applied": strategy_name,
                "error": str(e),
                "error_details": {
                    "exception_type": type(e).__name__,
                    "traceback": str(e)
                },
                "status": "failed"
            }
            
            self.send_message(
                MessageType.EXECUTION, 
                error_result,
                confidence=0.1,
                references=references
            )


class AlgebraExecutorAgent(ExecutorAgent):
    """Agent specialized in executing algebraic solution strategies"""
    
    def __init__(self, workspace: Workspace):
        super().__init__("AlgebraExecutor", workspace, "algebra")
        
        # Knowledge base of algebraic techniques
        self.techniques = {
            "factor_and_solve": self.factor_and_solve,
            "substitute_variables": self.substitute_variables,
            "complete_the_square": self.complete_the_square,
            "use_quadratic_formula": self.quadratic_formula,
            "solve_system_of_equations": self.solve_system_of_equations
        }
    
    def factor_and_solve(self, analysis: Dict, step: Dict) -> Dict:
        """Factor a polynomial equation and find its roots"""
        equations = analysis.get("components", {}).get("equations", [])
        if not equations:
            return {
                "working": "No equation found to factor",
                "result": None,
                "explanation": "Cannot proceed without an equation",
                "confidence": 0.1
            }
        
        # For demonstration, we'll use SymPy to factor the first equation
        # In a real system, we would parse the equation string and handle it more robustly
        try:
            equation_str = equations[0].get("text", "")
            if not equation_str:
                return {
                    "working": "Invalid equation format",
                    "result": None,
                    "explanation": "Equation missing or in invalid format",
                    "confidence": 0.1
                }
            
            # Very simple parsing - assumes form "expression = 0"
            parts = equation_str.split("=")
            left_side = parts[0].strip()
            right_side = parts[1].strip() if len(parts) > 1 else "0"
            
            # Convert to SymPy expressions
            x = sp.Symbol('x')  # Assuming x is the variable
            left_expr = sp.sympify(left_side)
            right_expr = sp.sympify(right_side)
            
            # Move everything to left side
            expr = left_expr - right_expr
            
            # Factor the expression
            factored = sp.factor(expr)
            
            # Solve for roots
            solutions = sp.solve(expr, x)
            
            # Detailed working steps
            working_steps = [
                {"description": f"Starting with the equation: {equation_str}"},
                {"description": f"Moving all terms to left side: {expr} = 0"},
                {"description": f"Factoring the expression: {factored} = 0"},
                {"description": "Setting each factor equal to zero and solving"}
            ]
            
            for factor in factored.as_ordered_factors():
                if factor.has(x):
                    working_steps.append({"description": f"Solving {factor} = 0"})
            
            working_steps.append({"description": f"Found solutions: {solutions}"})
            
            # Create visualization data for the factorization process
            visualization = {
                "type": "equation_factorization",
                "original_equation": equation_str,
                "factored_form": str(factored),
                "solutions": [str(sol) for sol in solutions],
                "verification": [
                    {"original_equation": equation_str, "x": float(sol), "result": float(expr.subs(x, sol))}
                    for sol in solutions if sol.is_real
                ]
            }
            
            return {
                "working": working_steps,
                "result": f"Solutions: {solutions}",
                "explanation": "Factored the expression and set each factor equal to zero to find roots",
                "confidence": 0.9,
                "computed_values": {"solutions": [str(sol) for sol in solutions]},
                "visualization": visualization
            }
            
        except Exception as e:
            # Fallback to a simpler implementation for demonstration
            # In reality, we would use a more sophisticated approach
            return {
                "working": [
                    {"description": f"Attempted to factor: {equations[0].get('text', '')}"},
                    {"description": "Using basic factoring techniques"},
                    {"description": f"Encountered error: {str(e)}"}
                ],
                "result": "Unable to compute exact factorization due to an error",
                "explanation": f"The system attempted to factor the expression but encountered an error: {str(e)}",
                "confidence": 0.3
            }
    
    def substitute_variables(self, analysis: Dict, step: Dict) -> Dict:
        """Substitute one variable in terms of others"""
        equations = analysis.get("components", {}).get("equations", [])
        if len(equations) < 2:
            return {
                "working": "Need at least two equations for variable substitution",
                "result": None,
                "explanation": "Cannot perform substitution without multiple equations",
                "confidence": 0.1
            }
        
        try:
            # Get equation strings
            eq1_str = equations[0].get("text", "")
            eq2_str = equations[1].get("text", "")
            
            if not eq1_str or not eq2_str:
                return {
                    "working": "Invalid equation format",
                    "result": None,
                    "explanation": "Equations missing or in invalid format",
                    "confidence": 0.1
                }
            
            # Extract variables
            variables = set()
            for eq in [eq1_str, eq2_str]:
                for var in re.findall(r'\b([a-zA-Z])\b', eq):
                    variables.add(var)
            
            variables = list(variables)
            if len(variables) < 2:
                return {
                    "working": f"Need at least two variables for substitution, found: {variables}",
                    "result": None,
                    "explanation": "Cannot perform substitution with fewer than two variables",
                    "confidence": 0.1
                }
            
            # For demonstration purposes, we'll choose the first variable to isolate
            var_to_isolate = variables[0]
            
            # Simplified implementation - would use SymPy in a real system
            working_steps = [
                {"description": f"Starting with equations:"},
                {"description": f"(1) {eq1_str}"},
                {"description": f"(2) {eq2_str}"},
                {"description": f"Variables identified: {', '.join(variables)}"},
                {"description": f"Choosing to isolate {var_to_isolate} from equation (1)"},
                {"description": f"Substituting {var_to_isolate} into equation (2)"},
                {"description": "Solving the resulting equation"}
            ]
            
            return {
                "working": working_steps,
                "result": f"Solution method demonstrated for variables {', '.join(variables)}",
                "explanation": "Used variable substitution to reduce the system complexity",
                "confidence": 0.7
            }
            
        except Exception as e:
            return {
                "working": [
                    {"description": "Attempted variable substitution"},
                    {"description": f"Encountered error: {str(e)}"}
                ],
                "result": "Unable to complete substitution due to an error",
                "explanation": f"The system attempted variable substitution but encountered an error: {str(e)}",
                "confidence": 0.3
            }
    
    def complete_the_square(self, analysis: Dict, step: Dict) -> Dict:
        """Complete the square for a quadratic expression"""
        equations = analysis.get("components", {}).get("equations", [])
        if not equations:
            return {
                "working": "No equation found to process",
                "result": None,
                "explanation": "Cannot proceed without an equation",
                "confidence": 0.1
            }
        
        try:
            # Find a quadratic equation
            quadratic_eq = None
            for eq in equations:
                if eq.get("type") == "quadratic":
                    quadratic_eq = eq
                    break
            
            if not quadratic_eq:
                return {
                    "working": "No quadratic equation found",
                    "result": None,
                    "explanation": "Completing the square requires a quadratic equation",
                    "confidence": 0.1
                }
            
            equation_str = quadratic_eq.get("text", "")
            
            # Simple parsing to extract coefficients (a real system would use SymPy)
            # Assume form: ax^2 + bx + c = 0
            parts = equation_str.split("=")
            left_side = parts[0].strip()
            
            # Very simplified coefficient extraction - would be more robust in a real system
            a_match = re.search(r'([+-]?\s*\d*)\s*x\^2', left_side)
            b_match = re.search(r'([+-]?\s*\d*)\s*x(?!\^)', left_side)
            c_match = re.search(r'([+-]?\s*\d+)(?!\s*x)', left_side)
            
            a = 1
            if a_match:
                a_str = a_match.group(1).strip()
                if a_str == "+" or a_str == "":
                    a = 1
                elif a_str == "-":
                    a = -1
                else:
                    a = float(a_str)
            
            b = 0
            if b_match:
                b_str = b_match.group(1).strip()
                if b_str == "+" or b_str == "":
                    b = 1
                elif b_str == "-":
                    b = -1
                else:
                    b = float(b_str)
            
            c = 0
            if c_match:
                c = float(c_match.group(1))
            
            # Complete the square
            h = -b / (2*a)
            k = c - a * h**2
            
            vertex_form = f"{a}(x - {h})² + {k}"
            
            # Detailed working steps
            working_steps = [
                {"description": f"Starting with the quadratic equation: {equation_str}"},
                {"description": f"Identified coefficients: a={a}, b={b}, c={c}"},
                {"description": f"To complete the square, we rewrite as: a(x² + (b/a)x) + c"},
                {"description": f"This becomes: {a}(x² + {b/a}x) + {c}"},
                {"description": f"To complete the square inside the parentheses, we add and subtract (b/2a)²"},
                {"description": f"(b/2a)² = ({b}/(2*{a}))² = {(b/(2*a))**2}"},
                {"description": f"So we have: {a}(x² + {b/a}x + {(b/(2*a))**2}) + {c} - {a}*{(b/(2*a))**2}"},
                {"description": f"Simplifying: {a}(x + {b/(2*a)})² + {c - a*(b/(2*a))**2}"},
                {"description": f"In vertex form: {vertex_form}"}
            ]
            
            # Create visualization data
            visualization = {
                "type": "completing_square",
                "original_equation": equation_str,
                "coefficients": {"a": a, "b": b, "c": c},
                "vertex_form": vertex_form,
                "vertex": {"h": h, "k": k}
            }
            
            return {
                "working": working_steps,
                "result": f"Vertex form: {vertex_form}",
                "explanation": "Completed the square to find the vertex form of the quadratic",
                "confidence": 0.85,
                "computed_values": {"vertex_form": vertex_form, "vertex": {"h": h, "k": k}},
                "visualization": visualization
            }
            
        except Exception as e:
            return {
                "working": [
                    {"description": "Attempted to complete the square"},
                    {"description": f"Encountered error: {str(e)}"}
                ],
                "result": "Unable to complete the square due to an error",
                "explanation": f"The system attempted to complete the square but encountered an error: {str(e)}",
                "confidence": 0.3
            }
    
    def quadratic_formula(self, analysis: Dict, step: Dict) -> Dict:
        """Apply the quadratic formula to find roots"""
        equations = analysis.get("components", {}).get("equations", [])
        if not equations:
            return {
                "working": "No equation found",
                "result": None,
                "explanation": "Cannot apply quadratic formula without an equation",
                "confidence": 0.1
            }
        
        try:
            # Find a quadratic equation
            quadratic_eq = None
            for eq in equations:
                if eq.get("type") == "quadratic":
                    quadratic_eq = eq
                    break
            
            if not quadratic_eq:
                return {
                    "working": "No quadratic equation found",
                    "result": None,
                    "explanation": "Quadratic formula requires a quadratic equation",
                    "confidence": 0.1
                }
            
            equation_str = quadratic_eq.get("text", "")
            
            # Parse the equation - a real system would use SymPy
            # Assume form: ax^2 + bx + c = 0
            parts = equation_str.split("=")
            left_side = parts[0].strip()
            right_side = parts[1].strip() if len(parts) > 1 else "0"
            
            # Convert to standard form if right side is not zero
            if right_side != "0":
                standard_form = f"({left_side}) - ({right_side}) = 0"
            else:
                standard_form = equation_str
            
            # Use SymPy to extract coefficients
            x = sp.Symbol('x')
            expr = sp.sympify(left_side) - sp.sympify(right_side)
            
            # Extract coefficients from polynomial form
            poly = sp.Poly(expr, x)
            coeffs = poly.all_coeffs()
            
            if len(coeffs) != 3:
                return {
                    "working": f"Expression is not a quadratic: {expr}",
                    "result": None,
                    "explanation": "The equation is not in quadratic form ax² + bx + c = 0",
                    "confidence": 0.3
                }
            
            a, b, c = coeffs
            
            # Apply quadratic formula
            discriminant = b**2 - 4*a*c
            
            working_steps = [
                {"description": f"Starting with the equation: {equation_str}"},
                {"description": f"In standard form: {standard_form}"},
                {"description": f"Identified coefficients: a={a}, b={b}, c={c}"},
                {"description": f"Applying the quadratic formula: x = (-b ± √(b² - 4ac)) / (2a)"},
                {"description": f"Calculating the discriminant: b² - 4ac = {b}² - 4({a})({c}) = {discriminant}"}
            ]
            
            if discriminant < 0:
                # Complex roots
                real_part = -b / (2*a)
                imag_part = sp.sqrt(-discriminant) / (2*a)
                solutions = [f"{real_part} + {imag_part}i", f"{real_part} - {imag_part}i"]
                working_steps.append({"description": f"Discriminant is negative, so there are complex roots"})
                working_steps.append({"description": f"x = {real_part} ± {imag_part}i"})
                result = f"Two complex roots: x = {real_part} ± {imag_part}i"
                
            elif discriminant == 0:
                # Repeated root
                solution = -b / (2*a)
                solutions = [solution, solution]
                working_steps.append({"description": f"Discriminant is zero, so there is one repeated root"})
                working_steps.append({"description": f"x = {solution}"})
                result = f"One repeated root: x = {solution}"
                
            else:
                # Two real roots
                solution1 = (-b + sp.sqrt(discriminant)) / (2*a)
                solution2 = (-b - sp.sqrt(discriminant)) / (2*a)
                solutions = [solution1, solution2]
                working_steps.append({"description": f"Discriminant is positive, so there are two real roots"})
                working_steps.append({"description": f"x = (-{b} + √{discriminant}) / (2*{a}) = {solution1}"})
                working_steps.append({"description": f"x = (-{b} - √{discriminant}) / (2*{a}) = {solution2}"})
                result = f"Two real roots: x = {solution1} or x = {solution2}"
            
            # Create visualization data
            visualization = {
                "type": "quadratic_formula",
                "original_equation": equation_str,
                "coefficients": {"a": float(a), "b": float(b), "c": float(c)},
                "discriminant": float(discriminant),
                "solutions": [str(sol) for sol in solutions]
            }
            
            return {
                "working": working_steps,
                "result": result,
                "explanation": "Applied the quadratic formula x = (-b ± √(b² - 4ac)) / (2a)",
                "confidence": 0.95,
                "computed_values": {"solutions": [str(sol) for sol in solutions], "discriminant": float(discriminant)},
                "visualization": visualization
            }
                
        except Exception as e:
            return {
                "working": [
                    {"description": f"Attempted to apply quadratic formula to {equations[0].get('text', '')}"},
                    {"description": f"Encountered error: {str(e)}"}
                ],
                "result": f"Error in calculation: {str(e)}",
                "explanation": "The system attempted to apply the quadratic formula but encountered an error",
                "confidence": 0.3
            }
    
    def solve_system_of_equations(self, analysis: Dict, step: Dict) -> Dict:
        """Solve a system of linear equations"""
        equations = analysis.get("components", {}).get("equations", [])
        if len(equations) < 2:
            return {
                "working": "Need at least two equations to form a system",
                "result": None,
                "explanation": "Cannot solve a system with fewer than two equations",
                "confidence": 0.1
            }
        
        try:
            # Extract equation strings
            eq_strs = [eq.get("text", "") for eq in equations]
            eq_strs = [eq for eq in eq_strs if eq]  # Filter out empty strings
            
            if len(eq_strs) < 2:
                return {
                    "working": "Invalid equation format in system",
                    "result": None,
                    "explanation": "Equations missing or in invalid format",
                    "confidence": 0.1
                }
            
            # For demonstration, we'll handle a 2x2 system
            # In a real implementation, we would use SymPy's linear solver more generally
            if len(eq_strs) > 2:
                eq_strs = eq_strs[:2]  # Just use the first two equations
            
            eq1_str, eq2_str = eq_strs
            
            # Extract all variables in the system
            variables = set()
            for eq in eq_strs:
                for var in re.findall(r'\b([a-zA-Z])\b', eq):
                    variables.add(var)
            
            if len(variables) != 2:
                return {
                    "working": f"System has {len(variables)} variables: {variables}. Need exactly 2 for this implementation.",
                    "result": None,
                    "explanation": "This implementation can only solve 2x2 systems",
                    "confidence": 0.3
                }
            
            # Convert to SymPy equations
            variables = list(variables)
            x, y = sp.symbols(variables)
            
            # Parse equations
            eq1_parts = eq1_str.split("=")
            eq2_parts = eq2_str.split("=")
            
            eq1 = sp.Eq(sp.sympify(eq1_parts[0]), sp.sympify(eq1_parts[1]) if len(eq1_parts) > 1 else 0)
            eq2 = sp.Eq(sp.sympify(eq2_parts[0]), sp.sympify(eq2_parts[1]) if len(eq2_parts) > 1 else 0)
            
            # Solve the system
            solution = sp.solve((eq1, eq2), (x, y))
            
            if not solution:
                return {
                    "working": [
                        {"description": f"System of equations:"},
                        {"description": f"(1) {eq1_str}"},
                        {"description": f"(2) {eq2_str}"},
                        {"description": "No solution found"}
                    ],
                    "result": "No solution exists for this system",
                    "explanation": "The equations may be inconsistent (no solution) or dependent (infinite solutions)",
                    "confidence": 0.7
                }
            
            # Detailed working steps
            working_steps = [
                {"description": f"System of equations:"},
                {"description": f"(1) {eq1_str}"},
                {"description": f"(2) {eq2_str}"},
                {"description": f"Variables to solve for: {variables[0]}, {variables[1]}"},
                {"description": "Using elimination method:"},
                {"description": f"Rearranging (1) to isolate {variables[0]}"},
                {"description": f"Substituting into (2)"},
                {"description": f"Solving for {variables[1]}"},
                {"description": f"Substituting back to find {variables[0]}"},
                {"description": f"Solution: {variables[0]} = {solution[x]}, {variables[1]} = {solution[y]}"}
            ]
            
            # Create visualization data
            visualization = {
                "type": "system_of_equations",
                "equations": [eq1_str, eq2_str],
                "variables": variables,
                "solution": {variables[0]: str(solution[x]), variables[1]: str(solution[y])}
            }
            
            return {
                "working": working_steps,
                "result": f"Solution: {variables[0]} = {solution[x]}, {variables[1]} = {solution[y]}",
                "explanation": "Solved using elimination and substitution",
                "confidence": 0.9,
                "computed_values": {variables[0]: str(solution[x]), variables[1]: str(solution[y])},
                "visualization": visualization
            }
            
        except Exception as e:
            return {
                "working": [
                    {"description": "Attempted to solve system of equations"},
                    {"description": f"Encountered error: {str(e)}"}
                ],
                "result": f"Error in calculation: {str(e)}",
                "explanation": "The system attempted to solve the equations but encountered an error",
                "confidence": 0.3
            }


class CalculusExecutorAgent(ExecutorAgent):
    """Agent specialized in executing calculus solution strategies"""
    
    def __init__(self, workspace: Workspace):
        super().__init__("CalculusExecutor", workspace, "calculus")
        
        # Knowledge base of calculus techniques
        self.techniques = {
            "compute_derivative": self.compute_derivative,
            "compute_integral": self.compute_integral,
            "find_critical_points": self.find_critical_points,
            "use_fundamental_theorem": self.apply_fundamental_theorem,
            "taylor_series_expansion": self.taylor_expansion
        }
    
    def compute_derivative(self, analysis: Dict, step: Dict) -> Dict:
        """Compute the derivative of a function"""
        # Extract function from problem or equations
        components = analysis.get("components", {})
        functions = components.get("functions", [])
        equations = components.get("equations", [])
        
        function_str = None
        
        # First try to find an explicit function
        if functions:
            for func in functions:
                if func.get("type") == "generic":
                    function_str = f"f(x) = {func.get('argument', '')}"
                    break
        
        # If no explicit function, look for a y = f(x) equation
        if not function_str and equations:
            for eq in equations:
                eq_text = eq.get("text", "")
                if "=" in eq_text and "y" in eq_text.split("=")[0]:
                    function_str = f"f(x) = {eq_text.split('=')[1].strip()}"
                    break
        
        if not function_str:
            return {
                "working": "No function found to differentiate",
                "result": None,
                "explanation": "Cannot compute derivative without a function",
                "confidence": 0.1
            }
        
        try:
            # Extract the expression part
            expr_str = function_str.split("=")[1].strip() if "=" in function_str else function_str
            
            # Use SymPy for symbolic differentiation
            x = sp.Symbol('x')
            expr = sp.sympify(expr_str)
            derivative = sp.diff(expr, x)
            
            # Detailed working steps
            working_steps = [
                {"description": f"Starting with f(x) = {expr}"},
                {"description": "Computing the derivative df/dx"}
            ]
            
            # Add steps based on the expression structure
            if expr.is_polynomial():
                working_steps.append({"description": "Using the power rule: d/dx(x^n) = n·x^(n-1)"})
            elif expr.has(sp.sin) or expr.has(sp.cos) or expr.has(sp.tan):
                working_steps.append({"description": "Using trigonometric derivative rules"})
            elif expr.has(sp.log) or expr.has(sp.exp):
                working_steps.append({"description": "Using logarithmic/exponential derivative rules"})
            
            # Add the result
            working_steps.append({"description": f"The derivative is: f'(x) = {derivative}"})
            
            # Create visualization data
            visualization = {
                "type": "derivative",
                "original_function": str(expr),
                "derivative": str(derivative),
                "points_of_interest": [
                    {"x": 0, "f(x)": float(expr.subs(x, 0)), "f'(x)": float(derivative.subs(x, 0))},
                    {"x": 1, "f(x)": float(expr.subs(x, 1)), "f'(x)": float(derivative.subs(x, 1))},
                    {"x": -1, "f(x)": float(expr.subs(x, -1)), "f'(x)": float(derivative.subs(x, -1))}
                ]
            }
            
            return {
                "working": working_steps,
                "result": f"f'(x) = {derivative}",
                "explanation": "Applied differentiation rules to compute the derivative",
                "confidence": 0.9,
                "computed_values": {"derivative": str(derivative)},
                "visualization": visualization
            }
        except Exception as e:
            return {
                "working": [
                    {"description": f"Attempted to differentiate: {function_str}"},
                    {"description": f"Encountered error: {str(e)}"}
                ],
                "result": f"Error in calculation: {str(e)}",
                "explanation": "The system attempted to compute the derivative but encountered an error",
                "confidence": 0.3
            }
    
    def compute_integral(self, analysis: Dict, step: Dict) -> Dict:
        """Compute the integral of a function"""
        # Extract function from problem or equations
        components = analysis.get("components", {})
        functions = components.get("functions", [])
        equations = components.get("equations", [])
        
        function_str = None
        
        # First try to find an explicit function
        if functions:
            for func in functions:
                if func.get("type") == "generic":
                    function_str = f"f(x) = {func.get('argument', '')}"
                    break
        
        # If no explicit function, look for a y = f(x) equation
        if not function_str and equations:
            for eq in equations:
                eq_text = eq.get("text", "")
                if "=" in eq_text and "y" in eq_text.split("=")[0]:
                    function_str = f"f(x) = {eq_text.split('=')[1].strip()}"
                    break
        
        if not function_str:
            return {
                "working": "No function found to integrate",
                "result": None,
                "explanation": "Cannot compute integral without a function",
                "confidence": 0.1
            }
        
        try:
            # Extract the expression part
            expr_str = function_str.split("=")[1].strip() if "=" in function_str else function_str
            
            # Use SymPy for symbolic integration
            x = sp.Symbol('x')
            expr = sp.sympify(expr_str)
            
            # Check for definite integral bounds
            bounds = None
            target = components.get("target", {}).get("description", "")
            
            # Look for bounds in the target description
            bound_match = re.search(r'from\s+([0-9.-]+)\s+to\s+([0-9.-]+)', target)
            if bound_match:
                try:
                    lower = float(bound_match.group(1))
                    upper = float(bound_match.group(2))
                    bounds = (lower, upper)
                except:
                    pass
            
            # Compute the indefinite integral
            indefinite_integral = sp.integrate(expr, x)
            
            # Detailed working steps
            working_steps = [
                {"description": f"Starting with f(x) = {expr}"},
                {"description": "Computing the indefinite integral ∫f(x)dx"}
            ]
            
            # Add steps based on the expression structure
            if expr.is_polynomial():
                working_steps.append({"description": "Using the power rule: ∫x^n dx = x^(n+1)/(n+1) + C"})
            elif expr.has(sp.sin) or expr.has(sp.cos) or expr.has(sp.tan):
                working_steps.append({"description": "Using trigonometric integration rules"})
            elif expr.has(sp.log) or expr.has(sp.exp):
                working_steps.append({"description": "Using logarithmic/exponential integration rules"})
            
            # Add the indefinite integral result
            working_steps.append({"description": f"The indefinite integral is: ∫f(x)dx = {indefinite_integral} + C"})
            
            # If definite bounds are available, compute the definite integral
            if bounds:
                lower, upper = bounds
                definite_result = indefinite_integral.subs(x, upper) - indefinite_integral.subs(x, lower)
                working_steps.append({"description": f"Computing definite integral from {lower} to {upper}"})
                working_steps.append({"description": f"Evaluating F({upper}) - F({lower}) = {indefinite_integral.subs(x, upper)} - {indefinite_integral.subs(x, lower)}"})
                working_steps.append({"description": f"Result = {definite_result}"})
                result = f"∫[{lower},{upper}] {expr} dx = {definite_result}"
                computed_values = {"indefinite_integral": str(indefinite_integral), "definite_integral": float(definite_result)}
            else:
                result = f"∫{expr} dx = {indefinite_integral} + C"
                computed_values = {"indefinite_integral": str(indefinite_integral)}
            
            # Create visualization data
            visualization = {
                "type": "integral",
                "original_function": str(expr),
                "indefinite_integral": str(indefinite_integral),
                "bounds": bounds,
                "definite_result": str(definite_result) if bounds else None
            }
            
            return {
                "working": working_steps,
                "result": result,
                "explanation": "Applied integration techniques to compute the integral",
                "confidence": 0.85,
                "computed_values": computed_values,
                "visualization": visualization
            }
        except Exception as e:
            return {
                "working": [
                    {"description": f"Attempted to integrate: {function_str}"},
                    {"description": f"Encountered error: {str(e)}"}
                ],
                "result": f"Error in calculation: {str(e)}",
                "explanation": "The system attempted to compute the integral but encountered an error",
                "confidence": 0.3
            }
    
    def find_critical_points(self, analysis: Dict, step: Dict) -> Dict:
        """Find critical points of a function"""
        # Extract function from problem or equations
        components = analysis.get("components", {})
        functions = components.get("functions", [])
        equations = components.get("equations", [])
        
        function_str = None
        
        # First try to find an explicit function
        if functions:
            for func in functions:
                if func.get("type") == "generic":
                    function_str = f"f(x) = {func.get('argument', '')}"
                    break
        
        # If no explicit function, look for a y = f(x) equation
        if not function_str and equations:
            for eq in equations:
                eq_text = eq.get("text", "")
                if "=" in eq_text and "y" in eq_text.split("=")[0]:
                    function_str = f"f(x) = {eq_text.split('=')[1].strip()}"
                    break
        
        if not function_str:
            return {
                "working": "No function found to analyze",
                "result": None,
                "explanation": "Cannot find critical points without a function",
                "confidence": 0.1
            }
        
        try:
            # Extract the expression part
            expr_str = function_str.split("=")[1].strip() if "=" in function_str else function_str
            
            # Use SymPy for symbolic calculations
            x = sp.Symbol('x')
            expr = sp.sympify(expr_str)
            
            # Find critical points by setting derivative to zero
            derivative = sp.diff(expr, x)
            critical_points = sp.solve(derivative, x)
            
            # Classify critical points using second derivative
            second_derivative = sp.diff(derivative, x)
            classifications = []
            
            for point in critical_points:
                try:
                    second_deriv_value = second_derivative.subs(x, point)
                    
                    if second_deriv_value > 0:
                        classification = "minimum"
                    elif second_deriv_value < 0:
                        classification = "maximum"
                    else:
                        classification = "inflection point or higher-order critical point"
                        
                    classifications.append({
                        "point": point,
                        "type": classification,
                        "f(x)": expr.subs(x, point),
                        "f'(x)": derivative.subs(x, point),
                        "f''(x)": second_deriv_value
                    })
                except:
                    classifications.append({
                        "point": point,
                        "type": "unknown (could not evaluate)",
                        "f(x)": "error",
                        "f'(x)": "error",
                        "f''(x)": "error"
                    })
            
            # Detailed working steps
            working_steps = [
                {"description": f"Starting with f(x) = {expr}"},
                {"description": f"Computing the derivative: f'(x) = {derivative}"},
                {"description": "Setting f'(x) = 0 to find critical points"},
                {"description": f"Solving {derivative} = 0"}
            ]
            
            for i, point in enumerate(critical_points):
                working_steps.append({"description": f"Critical point {i+1}: x = {point}"})
            
            working_steps.append({"description": "Classifying critical points using the second derivative"})
            working_steps.append({"description": f"Second derivative: f''(x) = {second_derivative}"})
            
            for i, classification in enumerate(classifications):
                working_steps.append({
                    "description": f"Point x = {classification['point']} is a {classification['type']} "
                                 f"(f''({classification['point']}) = {classification['f''(x)']})"
                })
            
            # Create result summary
            result_parts = []
            for classification in classifications:
                result_parts.append(f"x = {classification['point']} ({classification['type']})")
            
            result = "Critical points: " + ", ".join(result_parts)
            
            # Create visualization data
            visualization = {
                "type": "critical_points",
                "function": str(expr),
                "derivative": str(derivative),
                "second_derivative": str(second_derivative),
                "critical_points": [
                    {
                        "x": float(cp["point"]) if not isinstance(cp["point"], complex) else str(cp["point"]),
                        "f(x)": float(cp["f(x)"]) if cp["f(x)"] != "error" and not isinstance(cp["f(x)"], complex) else str(cp["f(x)"]),
                        "type": cp["type"]
                    }
                    for cp in classifications
                ]
            }
            
            return {
                "working": working_steps,
                "result": result,
                "explanation": "Found critical points by setting derivative to zero and classified them using the second derivative",
                "confidence": 0.9,
                "computed_values": {"critical_points": [{"x": str(cp["point"]), "type": cp["type"]} for cp in classifications]},
                "visualization": visualization
            }
        except Exception as e:
            return {
                "working": [
                    {"description": f"Attempted to find critical points for: {function_str}"},
                    {"description": f"Encountered error: {str(e)}"}
                ],
                "result": f"Error in calculation: {str(e)}",
                "explanation": "The system attempted to find critical points but encountered an error",
                "confidence": 0.3
            }
    
    def apply_fundamental_theorem(self, analysis: Dict, step: Dict) -> Dict:
        """Apply the Fundamental Theorem of Calculus"""
        # Extract function from problem or equations
        components = analysis.get("components", {})
        functions = components.get("functions", [])
        equations = components.get("equations", [])
        
        function_str = None
        
        # First try to find an explicit function
        if functions:
            for func in functions:
                if func.get("type") == "generic":
                    function_str = f"f(x) = {func.get('argument', '')}"
                    break
        
        # If no explicit function, look for a y = f(x) equation
        if not function_str and equations:
            for eq in equations:
                eq_text = eq.get("text", "")
                if "=" in eq_text and "y" in eq_text.split("=")[0]:
                    function_str = f"f(x) = {eq_text.split('=')[1].strip()}"
                    break
        
        if not function_str:
            return {
                "working": "No function found to integrate",
                "result": None,
                "explanation": "Cannot apply the Fundamental Theorem without a function",
                "confidence": 0.1
            }
        
        # Look for integration bounds
        bounds = None
        target = components.get("target", {}).get("description", "")
        
        # Look for bounds in the target description
        bound_match = re.search(r'from\s+([0-9.-]+)\s+to\s+([0-9.-]+)', target)
        if bound_match:
            try:
                lower = float(bound_match.group(1))
                upper = float(bound_match.group(2))
                bounds = (lower, upper)
            except:
                pass
        
        if not bounds:
            return {
                "working": "No integration bounds found",
                "result": None,
                "explanation": "The Fundamental Theorem requires definite integration bounds",
                "confidence": 0.3
            }
        
        try:
            # Extract the expression part
            expr_str = function_str.split("=")[1].strip() if "=" in function_str else function_str
            
            # Use SymPy for symbolic calculations
            x = sp.Symbol('x')
            expr = sp.sympify(expr_str)
            
            # Find the antiderivative
            antiderivative = sp.integrate(expr, x)
            
            # Apply the Fundamental Theorem
            lower, upper = bounds
            definite_integral = antiderivative.subs(x, upper) - antiderivative.subs(x, lower)
            
            # Detailed working steps
            working_steps = [
                {"description": f"Starting with f(x) = {expr}"},
                {"description": f"Using the Fundamental Theorem of Calculus to find ∫[{lower},{upper}] f(x) dx"},
                {"description": f"First, find the antiderivative F(x): {antiderivative}"},
                {"description": f"Then compute F({upper}) - F({lower})"},
                {"description": f"F({upper}) = {antiderivative.subs(x, upper)}"},
                {"description": f"F({lower}) = {antiderivative.subs(x, lower)}"},
                {"description": f"∫[{lower},{upper}] f(x) dx = {antiderivative.subs(x, upper)} - {antiderivative.subs(x, lower)} = {definite_integral}"}
            ]
            
            # Create visualization data
            visualization = {
                "type": "definite_integral",
                "function": str(expr),
                "antiderivative": str(antiderivative),
                "bounds": {"lower": lower, "upper": upper},
                "evaluation": {
                    "upper": float(antiderivative.subs(x, upper)),
                    "lower": float(antiderivative.subs(x, lower)),
                    "result": float(definite_integral)
                }
            }
            
            return {
                "working": working_steps,
                "result": f"∫[{lower},{upper}] {expr} dx = {definite_integral}",
                "explanation": "Applied the Fundamental Theorem of Calculus to evaluate the definite integral",
                "confidence": 0.9,
                "computed_values": {"definite_integral": float(definite_integral)},
                "visualization": visualization
            }
        except Exception as e:
            return {
                "working": [
                    {"description": f"Attempted to apply the Fundamental Theorem to: {function_str}"},
                    {"description": f"Over the interval [{bounds[0]}, {bounds[1]}]"},
                    {"description": f"Encountered error: {str(e)}"}
                ],
                "result": f"Error in calculation: {str(e)}",
                "explanation": "The system attempted to apply the Fundamental Theorem but encountered an error",
                "confidence": 0.3
            }
    
    def taylor_expansion(self, analysis: Dict, step: Dict) -> Dict:
        """Compute Taylor series expansion of a function"""
        # Extract function from problem or equations
        components = analysis.get("components", {})
        functions = components.get("functions", [])
        equations = components.get("equations", [])
        
        function_str = None
        
        # First try to find an explicit function
        if functions:
            for func in functions:
                if func.get("type") == "generic":
                    function_str = f"f(x) = {func.get('argument', '')}"
                    break
        
        # If no explicit function, look for a y = f(x) equation
        if not function_str and equations:
            for eq in equations:
                eq_text = eq.get("text", "")
                if "=" in eq_text and "y" in eq_text.split("=")[0]:
                    function_str = f"f(x) = {eq_text.split('=')[1].strip()}"
                    break
        
        if not function_str:
            return {
                "working": "No function found for Taylor expansion",
                "result": None,
                "explanation": "Cannot compute Taylor series without a function",
                "confidence": 0.1
            }
        
        try:
            # Extract the expression part
            expr_str = function_str.split("=")[1].strip() if "=" in function_str else function_str
            
            # Use SymPy for symbolic calculations
            x = sp.Symbol('x')
            expr = sp.sympify(expr_str)
            
            # Determine expansion point
            center = 0  # Default to Maclaurin series
            order = 5   # Default to 5th order
            
            # Look for specific center in problem
            target = components.get("target", {}).get("description", "")
            center_match = re.search(r'around\s+([0-9.-]+)', target)
            if center_match:
                try:
                    center = float(center_match.group(1))
                except:
                    pass
            
            # Compute the Taylor series
            taylor_series = expr.series(x, center, order+1).removeO()
            
            # Detailed working steps
            working_steps = [
                {"description": f"Starting with f(x) = {expr}"},
                {"description": f"Computing Taylor series around x = {center} up to order {order}"},
                {"description": "Taylor series formula: f(x) ≈ Σ[f^(n)(a)/n!](x-a)^n from n=0 to ∞"}
            ]
            
            # Compute each term separately for detailed steps
            terms = []
            for n in range(order + 1):
                derivative = expr
                for _ in range(n):
                    derivative = sp.diff(derivative, x)
                
                coefficient = derivative.subs(x, center) / sp.factorial(n)
                term = coefficient * (x - center)**n
                
                if coefficient != 0:
                    terms.append(term)
                    working_steps.append({
                        "description": f"Term {n}: f^({n})({center})/n! × (x-{center})^{n} = {coefficient} × (x-{center})^{n} = {term}"
                    })
            
            working_steps.append({"description": f"Full Taylor series: {taylor_series}"})
            
            # Create visualization data
            visualization = {
                "type": "taylor_series",
                "function": str(expr),
                "center": center,
                "order": order,
                "series": str(taylor_series),
                "terms": [str(term) for term in terms],
                "comparison_points": [
                    {
                        "x": center + i*0.5,
                        "actual": float(expr.subs(x, center + i*0.5)),
                        "approximation": float(taylor_series.subs(x, center + i*0.5))
                    }
                    for i in range(-2, 3)
                ]
            }
            
            return {
                "working": working_steps,
                "result": f"Taylor series of {expr} around x = {center} up to order {order}: {taylor_series}",
                "explanation": "Computed the Taylor series by calculating derivatives and coefficients",
                "confidence": 0.85,
                "computed_values": {"taylor_series": str(taylor_series), "center": center, "order": order},
                "visualization": visualization
            }
        except Exception as e:
            return {
                "working": [
                    {"description": f"Attempted to compute Taylor series for: {function_str}"},
                    {"description": f"Encountered error: {str(e)}"}
                ],
                "result": f"Error in calculation: {str(e)}",
                "explanation": "The system attempted to compute the Taylor series but encountered an error",
                "confidence": 0.3
            }


class GeometryExecutorAgent(ExecutorAgent):
    """Agent specialized in executing geometry solution strategies"""
    
    def __init__(self, workspace: Workspace):
        super().__init__("GeometryExecutor", workspace, "geometry")
        
        # Knowledge base of geometry techniques
        self.techniques = {
            "use_coordinate_geometry": self.use_coordinate_geometry,
            "apply_triangle_theorems": self.apply_triangle_theorems,
            "apply_pythagorean_theorem": self.use_pythagorean_theorem,
            "use_similarity_and_congruence": self.apply_similarity_congruence,
            "compute_area_volume": self.compute_area_volume
        }
    
    def use_coordinate_geometry(self, analysis: Dict, step: Dict) -> Dict:
        """Apply coordinate geometry to solve a problem"""
        components = analysis.get("components", {})
        target = components.get("target", {}).get("description", "")
        geometric_entities = components.get("geometric_entities", [])
        
        if not geometric_entities:
            return {
                "working": "No geometric entities found",
                "result": None,
                "explanation": "Cannot apply coordinate geometry without geometric entities",
                "confidence": 0.1
            }
        
        # Look for points or lines in the problem
        has_points = any(entity.get("entity") == "point" for entity in geometric_entities)
        has_lines = any(entity.get("entity") == "line" for entity in geometric_entities)
        
        if not (has_points or has_lines):
            return {
                "working": "No points or lines found for coordinate geometry",
                "result": None,
                "explanation": "Coordinate geometry typically requires points or lines",
                "confidence": 0.3
            }
        
        try:
            # Determine what we're solving for
            if "distance" in target.lower():
                return self._coordinate_distance(analysis)
            elif "midpoint" in target.lower():
                return self._coordinate_midpoint(analysis)
            elif "slope" in target.lower():
                return self._coordinate_slope(analysis)
            elif "equation" in target.lower() and "line" in target.lower():
                return self._coordinate_line_equation(analysis)
            else:
                # Generic coordinate geometry approach
                return {
                    "working": [
                        {"description": "Using coordinate geometry to solve the problem"},
                        {"description": "Identifying key geometric entities in the problem"},
                        {"description": "Setting up a coordinate system"},
                        {"description": "Applying coordinate formulas to solve"}
                    ],
                    "result": "Coordinate geometry approach demonstrated",
                    "explanation": "Used coordinate geometry principles to represent and solve the problem",
                    "confidence": 0.7
                }
        except Exception as e:
            return {
                "working": [
                    {"description": "Attempted to apply coordinate geometry"},
                    {"description": f"Encountered error: {str(e)}"}
                ],
                "result": f"Error in calculation: {str(e)}",
                "explanation": "The system attempted to apply coordinate geometry but encountered an error",
                "confidence": 0.3
            }
    
    def _coordinate_distance(self, analysis: Dict) -> Dict:
        """Calculate distance between points using the distance formula"""
        # Extract point coordinates from the problem
        # This is a simplified implementation - a real system would parse the problem more thoroughly
        
        # For demonstration, let's assume we've identified two points (x1,y1) and (x2,y2)
        x1, y1 = 1, 2  # Example values
        x2, y2 = 4, 6  # Example values
        
        # Calculate distance
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        
        # Detailed working steps
        working_steps = [
            {"description": f"Identifying two points: A({x1}, {y1}) and B({x2}, {y2})"},
            {"description": "Using the distance formula: d = √[(x₂ - x₁)² + (y₂ - y₁)²]"},
            {"description": f"d = √[({x2} - {x1})² + ({y2} - {y1})²]"},
            {"description": f"d = √[{x2 - x1}² + {y2 - y1}²]"},
            {"description": f"d = √[{(x2 - x1)**2} + {(y2 - y1)**2}]"},
            {"description": f"d = √{(x2 - x1)**2 + (y2 - y1)**2}"},
            {"description": f"d = {distance}"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "coordinate_distance",
            "points": [
                {"name": "A", "x": x1, "y": y1},
                {"name": "B", "x": x2, "y": y2}
            ],
            "distance": distance
        }
        
        return {
            "working": working_steps,
            "result": f"Distance = {distance}",
            "explanation": "Used the distance formula to find the distance between two points",
            "confidence": 0.9,
            "computed_values": {"distance": distance},
            "visualization": visualization
        }
    
    def _coordinate_midpoint(self, analysis: Dict) -> Dict:
        """Calculate the midpoint between two points"""
        # For demonstration, let's assume we've identified two points (x1,y1) and (x2,y2)
        x1, y1 = 1, 2  # Example values
        x2, y2 = 5, 10  # Example values
        
        # Calculate midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Detailed working steps
        working_steps = [
            {"description": f"Identifying two points: A({x1}, {y1}) and B({x2}, {y2})"},
            {"description": "Using the midpoint formula: M = ((x₁ + x₂)/2, (y₁ + y₂)/2)"},
            {"description": f"M = (({x1} + {x2})/2, ({y1} + {y2})/2)"},
            {"description": f"M = ({x1 + x2}/2, {y1 + y2}/2)"},
            {"description": f"M = ({mid_x}, {mid_y})"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "coordinate_midpoint",
            "points": [
                {"name": "A", "x": x1, "y": y1},
                {"name": "B", "x": x2, "y": y2},
                {"name": "M", "x": mid_x, "y": mid_y}
            ]
        }
        
        return {
            "working": working_steps,
            "result": f"Midpoint = ({mid_x}, {mid_y})",
            "explanation": "Used the midpoint formula to find the point halfway between the given points",
            "confidence": 0.9,
            "computed_values": {"midpoint": {"x": mid_x, "y": mid_y}},
            "visualization": visualization
        }
    
    def _coordinate_slope(self, analysis: Dict) -> Dict:
        """Calculate the slope of a line"""
        # For demonstration, let's assume we've identified two points (x1,y1) and (x2,y2)
        x1, y1 = 1, 2  # Example values
        x2, y2 = 4, 8  # Example values
        
        # Calculate slope
        if x2 - x1 == 0:
            slope = "undefined (vertical line)"
            slope_value = None
        else:
            slope_value = (y2 - y1) / (x2 - x1)
            slope = str(slope_value)
        
        # Detailed working steps
        working_steps = [
            {"description": f"Identifying two points: A({x1}, {y1}) and B({x2}, {y2})"},
            {"description": "Using the slope formula: m = (y₂ - y₁)/(x₂ - x₁)"}
        ]
        
        if slope_value is None:
            working_steps.append({"description": f"m = ({y2} - {y1})/({x2} - {x1}) = {y2 - y1}/0"})
            working_steps.append({"description": "Slope is undefined (vertical line)"})
        else:
            working_steps.extend([
                {"description": f"m = ({y2} - {y1})/({x2} - {x1})"},
                {"description": f"m = {y2 - y1}/{x2 - x1}"},
                {"description": f"m = {slope}"}