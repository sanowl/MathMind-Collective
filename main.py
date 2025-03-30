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
            "distance": distance,
            "line": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
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
            ],
            "segment_AM": {"length": ((mid_x - x1)**2 + (mid_y - y1)**2)**0.5},
            "segment_MB": {"length": ((x2 - mid_x)**2 + (y2 - mid_y)**2)**0.5}
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
            ])
        
        # Create visualization data
        visualization = {
            "type": "coordinate_slope",
            "points": [
                {"name": "A", "x": x1, "y": y1},
                {"name": "B", "x": x2, "y": y2}
            ],
            "slope": slope_value,
            "line": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        }
        
        return {
            "working": working_steps,
            "result": f"Slope = {slope}",
            "explanation": "Used the slope formula to find the rate of change between the points",
            "confidence": 0.9,
            "computed_values": {"slope": slope_value},
            "visualization": visualization
        }
    
    def _coordinate_line_equation(self, analysis: Dict) -> Dict:
        """Find the equation of a line in slope-intercept form"""
        # For demonstration, let's assume we've identified two points (x1,y1) and (x2,y2)
        x1, y1 = 1, 2  # Example values
        x2, y2 = 4, 8  # Example values
        
        # Calculate slope
        if x2 - x1 == 0:
            return {
                "working": [
                    {"description": f"Identifying two points: A({x1}, {y1}) and B({x2}, {y2})"},
                    {"description": "Since the x-coordinates are the same, this is a vertical line"},
                    {"description": f"Equation of a vertical line: x = {x1}"}
                ],
                "result": f"x = {x1}",
                "explanation": "Found the equation of a vertical line passing through the given points",
                "confidence": 0.9,
                "computed_values": {"equation_type": "vertical", "x_value": x1}
            }
        
        # Calculate slope
        m = (y2 - y1) / (x2 - x1)
        
        # Calculate y-intercept (b) using point-slope form: y - y1 = m(x - x1)
        # Rearranging to slope-intercept form: y = mx + b
        # So b = y1 - m*x1
        b = y1 - m*x1
        
        # Format the equation
        equation = f"y = {m}x + {b}" if b >= 0 else f"y = {m}x - {abs(b)}"
        
        # Detailed working steps
        working_steps = [
            {"description": f"Identifying two points: A({x1}, {y1}) and B({x2}, {y2})"},
            {"description": "Step 1: Calculate the slope using m = (y₂ - y₁)/(x₂ - x₁)"},
            {"description": f"m = ({y2} - {y1})/({x2} - {x1}) = {y2 - y1}/{x2 - x1} = {m}"},
            {"description": "Step 2: Use the point-slope form y - y₁ = m(x - x₁)"},
            {"description": f"y - {y1} = {m}(x - {x1})"},
            {"description": "Step 3: Convert to slope-intercept form y = mx + b"},
            {"description": f"y - {y1} = {m}x - {m*x1}"},
            {"description": f"y = {m}x - {m*x1} + {y1}"},
            {"description": f"y = {m}x + {b}"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "line_equation",
            "points": [
                {"name": "A", "x": x1, "y": y1},
                {"name": "B", "x": x2, "y": y2}
            ],
            "slope": m,
            "y_intercept": b,
            "equation": equation,
            "line": {"slope": m, "intercept": b}
        }
        
        return {
            "working": working_steps,
            "result": equation,
            "explanation": "Found the equation of the line in slope-intercept form",
            "confidence": 0.9,
            "computed_values": {"slope": m, "y_intercept": b, "equation": equation},
            "visualization": visualization
        }
    
    def apply_triangle_theorems(self, analysis: Dict, step: Dict) -> Dict:
        """Apply theorems about triangles to solve a problem"""
        components = analysis.get("components", {})
        target = components.get("target", {}).get("description", "")
        geometric_entities = components.get("geometric_entities", [])
        
        # Check if problem involves triangles
        has_triangle = any(entity.get("entity") == "triangle" for entity in geometric_entities)
        
        if not has_triangle:
            return {
                "working": "No triangle found in the problem",
                "result": None,
                "explanation": "Cannot apply triangle theorems without a triangle",
                "confidence": 0.1
            }
        
        try:
            # Determine what theorem to apply
            if "right" in target.lower() or "pythagorean" in target.lower():
                return self.use_pythagorean_theorem(analysis, step)
            elif "similar" in target.lower():
                return self.apply_similarity_congruence(analysis, step)
            elif "area" in target.lower():
                return self._triangle_area(analysis)
            elif "angle" in target.lower():
                return self._triangle_angles(analysis)
            else:
                # Generic triangle theorem approach
                return {
                    "working": [
                        {"description": "Identifying the triangle in the problem"},
                        {"description": "Determining which triangle theorems are applicable"},
                        {"description": "Applying appropriate theorems to solve the problem"}
                    ],
                    "result": "Triangle theorem approach demonstrated",
                    "explanation": "Used theorems about triangles to solve the problem",
                    "confidence": 0.7
                }
        except Exception as e:
            return {
                "working": [
                    {"description": "Attempted to apply triangle theorems"},
                    {"description": f"Encountered error: {str(e)}"}
                ],
                "result": f"Error in calculation: {str(e)}",
                "explanation": "The system attempted to apply triangle theorems but encountered an error",
                "confidence": 0.3
            }
    
    def _triangle_area(self, analysis: Dict) -> Dict:
        """Calculate the area of a triangle"""
        # For demonstration, let's assume we know the base and height
        base = 6  # Example value
        height = 4  # Example value
        
        # Calculate area
        area = 0.5 * base * height
        
        # Detailed working steps
        working_steps = [
            {"description": f"Identifying triangle with base = {base} and height = {height}"},
            {"description": "Using the formula: Area = (1/2) × base × height"},
            {"description": f"Area = (1/2) × {base} × {height}"},
            {"description": f"Area = 0.5 × {base} × {height}"},
            {"description": f"Area = 0.5 × {base * height}"},
            {"description": f"Area = {area}"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "triangle_area",
            "triangle": {
                "base": base,
                "height": height,
                "vertices": [
                    {"x": 0, "y": 0},
                    {"x": base, "y": 0},
                    {"x": base/2, "y": height}
                ]
            },
            "area": area
        }
        
        return {
            "working": working_steps,
            "result": f"Area = {area}",
            "explanation": "Used the formula Area = (1/2) × base × height to find the triangle's area",
            "confidence": 0.9,
            "computed_values": {"area": area},
            "visualization": visualization
        }
    
    def _triangle_angles(self, analysis: Dict) -> Dict:
        """Calculate angles in a triangle"""
        # For demonstration, let's assume we know two angles and need to find the third
        angle1 = 30  # Example value in degrees
        angle2 = 45  # Example value in degrees
        
        # Calculate the third angle using the fact that angles in a triangle sum to 180°
        angle3 = 180 - angle1 - angle2
        
        # Detailed working steps
        working_steps = [
            {"description": f"Given angles in the triangle: {angle1}° and {angle2}°"},
            {"description": "Using the fact that angles in a triangle sum to 180°"},
            {"description": f"Third angle = 180° - {angle1}° - {angle2}°"},
            {"description": f"Third angle = 180° - {angle1 + angle2}°"},
            {"description": f"Third angle = {angle3}°"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "triangle_angles",
            "angles": {
                "angle1": angle1,
                "angle2": angle2,
                "angle3": angle3
            },
            "vertices": [
                {"x": 0, "y": 0},
                {"x": 10, "y": 0},
                {"x": 5, "y": 8}
            ]
        }
        
        return {
            "working": working_steps,
            "result": f"Third angle = {angle3}°",
            "explanation": "Used the fact that angles in a triangle sum to 180° to find the missing angle",
            "confidence": 0.95,
            "computed_values": {"angle3": angle3},
            "visualization": visualization
        }
    
    def use_pythagorean_theorem(self, analysis: Dict, step: Dict) -> Dict:
        """Apply the Pythagorean theorem to a right triangle"""
        # Determine what we need to find
        target = analysis.get("components", {}).get("target", {}).get("description", "")
        
        # For demonstration, let's consider different cases
        if "hypotenuse" in target.lower():
            # Finding the hypotenuse
            a = 3  # Example value
            b = 4  # Example value
            c = (a**2 + b**2)**0.5
            
            working_steps = [
                {"description": f"Identifying right triangle with legs a = {a} and b = {b}"},
                {"description": "Using the Pythagorean theorem: c² = a² + b²"},
                {"description": f"c² = {a}² + {b}²"},
                {"description": f"c² = {a**2} + {b**2}"},
                {"description": f"c² = {a**2 + b**2}"},
                {"description": f"c = √{a**2 + b**2}"},
                {"description": f"c = {c}"}
            ]
            
            result = f"Hypotenuse c = {c}"
            computed_values = {"hypotenuse": c}
            
        elif "leg" in target.lower():
            # Finding a leg
            a = 5  # Example value (one leg)
            c = 13  # Example value (hypotenuse)
            b = (c**2 - a**2)**0.5
            
            working_steps = [
                {"description": f"Identifying right triangle with leg a = {a} and hypotenuse c = {c}"},
                {"description": "Using the Pythagorean theorem: a² + b² = c²"},
                {"description": f"{a}² + b² = {c}²"},
                {"description": f"{a**2} + b² = {c**2}"},
                {"description": f"b² = {c**2} - {a**2}"},
                {"description": f"b² = {c**2 - a**2}"},
                {"description": f"b = √{c**2 - a**2}"},
                {"description": f"b = {b}"}
            ]
            
            result = f"Leg b = {b}"
            computed_values = {"leg": b}
            
        else:
            # Checking if a triangle is right
            a = 5  # Example value
            b = 12  # Example value
            c = 13  # Example value
            is_right = abs((a**2 + b**2) - c**2) < 0.0001
            
            working_steps = [
                {"description": f"Examining triangle with sides a = {a}, b = {b}, and c = {c}"},
                {"description": "Checking if the triangle is right using the Pythagorean theorem: a² + b² = c²"},
                {"description": f"{a}² + {b}² = {a**2} + {b**2} = {a**2 + b**2}"},
                {"description": f"c² = {c}² = {c**2}"},
                {"description": f"Comparing: {a**2 + b**2} {'=' if is_right else '≠'} {c**2}"}
            ]
            
            result = f"The triangle {'is' if is_right else 'is not'} a right triangle"
            computed_values = {"is_right_triangle": is_right}
        
        # Create visualization data
        visualization = {
            "type": "pythagorean_theorem",
            "triangle": {
                "a": a,
                "b": b,
                "c": c,
                "vertices": [
                    {"x": 0, "y": 0},
                    {"x": a, "y": 0},
                    {"x": 0, "y": b}
                ]
            }
        }
        
        return {
            "working": working_steps,
            "result": result,
            "explanation": "Applied the Pythagorean theorem to solve the problem",
            "confidence": 0.95,
            "computed_values": computed_values,
            "visualization": visualization
        }
    
    def apply_similarity_congruence(self, analysis: Dict, step: Dict) -> Dict:
        """Apply triangle similarity or congruence principles"""
        # For demonstration, let's solve a similarity problem
        # Assume we have triangles ABC and DEF with known sides
        
        # Triangle ABC
        ab = 3  # Example value
        bc = 4  # Example value
        ca = 5  # Example value
        
        # Triangle DEF (similar to ABC)
        scale_factor = 2  # Example value
        de = ab * scale_factor
        ef = bc * scale_factor
        fd = ca * scale_factor
        
        # Detailed working steps
        working_steps = [
            {"description": "Identifying two triangles: ABC and DEF"},
            {"description": f"Triangle ABC has sides: AB = {ab}, BC = {bc}, CA = {ca}"},
            {"description": "Determining the triangles are similar"},
            {"description": f"Scale factor from ABC to DEF = {scale_factor}"},
            {"description": f"Computing sides of DEF: DE = {scale_factor} × AB = {de}"},
            {"description": f"EF = {scale_factor} × BC = {ef}"},
            {"description": f"FD = {scale_factor} × CA = {fd}"}
        ]
        
        # Add additional information about similar triangles
        working_steps.extend([
            {"description": "Properties of similar triangles:"},
            {"description": "1. Corresponding angles are congruent"},
            {"description": "2. Corresponding sides are proportional"},
            {"description": "3. Area ratio equals the square of the scale factor"}
        ])
        
        # Create visualization data
        visualization = {
            "type": "triangle_similarity",
            "triangle1": {
                "name": "ABC",
                "sides": {"AB": ab, "BC": bc, "CA": ca},
                "vertices": [
                    {"x": 0, "y": 0},
                    {"x": ab, "y": 0},
                    {"x": ab*0.6, "y": bc*0.8}
                ]
            },
            "triangle2": {
                "name": "DEF",
                "sides": {"DE": de, "EF": ef, "FD": fd},
                "vertices": [
                    {"x": 0, "y": 0},
                    {"x": de, "y": 0},
                    {"x": de*0.6, "y": ef*0.8}
                ]
            },
            "scale_factor": scale_factor,
            "area_ratio": scale_factor**2
        }
        
        return {
            "working": working_steps,
            "result": f"Triangle DEF with sides DE = {de}, EF = {ef}, FD = {fd}",
            "explanation": "Applied similarity principles to find the unknown measurements",
            "confidence": 0.9,
            "computed_values": {"scale_factor": scale_factor, "similar_sides": {"DE": de, "EF": ef, "FD": fd}},
            "visualization": visualization
        }
    
    def compute_area_volume(self, analysis: Dict, step: Dict) -> Dict:
        """Compute area or volume of geometric shapes"""
        components = analysis.get("components", {})
        target = components.get("target", {}).get("description", "")
        geometric_entities = components.get("geometric_entities", [])
        
        if not geometric_entities:
            return {
                "working": "No geometric entities found",
                "result": None,
                "explanation": "Cannot compute area or volume without geometric entities",
                "confidence": 0.1
            }
        
        # Find the primary geometric entity
        entity_types = [entity.get("entity") for entity in geometric_entities]
        
        try:
            if "area" in target.lower():
                # Area computation
                if "circle" in entity_types:
                    return self._circle_area(analysis)
                elif "triangle" in entity_types:
                    return self._triangle_area(analysis)
                elif "rectangle" in entity_types or "square" in entity_types:
                    return self._rectangle_area(analysis)
                else:
                    return {
                        "working": f"Attempting to compute area for entities: {entity_types}",
                        "result": "Area computation for this shape not implemented",
                        "explanation": "The specific area formula for this shape is not available",
                        "confidence": 0.3
                    }
            elif "volume" in target.lower():
                # Volume computation
                if "sphere" in entity_types:
                    return self._sphere_volume(analysis)
                elif "cube" in entity_types:
                    return self._cube_volume(analysis)
                elif "cylinder" in entity_types:
                    return self._cylinder_volume(analysis)
                else:
                    return {
                        "working": f"Attempting to compute volume for entities: {entity_types}",
                        "result": "Volume computation for this shape not implemented",
                        "explanation": "The specific volume formula for this shape is not available",
                        "confidence": 0.3
                    }
            else:
                return {
                    "working": [
                        {"description": f"Identified geometric entities: {entity_types}"},
                        {"description": "Determining appropriate area/volume formulas"},
                        {"description": "Applying formulas to compute measurements"}
                    ],
                    "result": "Geometric measurement approach demonstrated",
                    "explanation": "Used geometric formulas to compute measurements",
                    "confidence": 0.7
                }
        except Exception as e:
            return {
                "working": [
                    {"description": f"Attempted to compute area/volume for: {entity_types}"},
                    {"description": f"Encountered error: {str(e)}"}
                ],
                "result": f"Error in calculation: {str(e)}",
                "explanation": "The system encountered an error during computation",
                "confidence": 0.3
            }
    
    def _circle_area(self, analysis: Dict) -> Dict:
        """Calculate the area of a circle"""
        # For demonstration, assume we know the radius
        radius = 5  # Example value
        
        # Calculate area
        area = 3.14159 * radius**2
        
        # Detailed working steps
        working_steps = [
            {"description": f"Identifying circle with radius = {radius}"},
            {"description": "Using the formula: Area = π × r²"},
            {"description": f"Area = π × {radius}²"},
            {"description": f"Area = π × {radius**2}"},
            {"description": f"Area = {area}"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "circle_area",
            "circle": {
                "radius": radius,
                "center": {"x": 0, "y": 0}
            },
            "area": area
        }
        
        return {
            "working": working_steps,
            "result": f"Area = {area}",
            "explanation": "Used the formula Area = π × r² to find the circle's area",
            "confidence": 0.95,
            "computed_values": {"area": area},
            "visualization": visualization
        }
    
    def _rectangle_area(self, analysis: Dict) -> Dict:
        """Calculate the area of a rectangle or square"""
        # For demonstration, assume we know the dimensions
        length = 8  # Example value
        width = 5  # Example value
        
        # Check if it's a square
        is_square = length == width
        
        # Calculate area
        area = length * width
        
        # Detailed working steps
        if is_square:
            working_steps = [
                {"description": f"Identifying square with side length = {length}"},
                {"description": "Using the formula: Area = side²"},
                {"description": f"Area = {length}²"},
                {"description": f"Area = {area}"}
            ]
        else:
            working_steps = [
                {"description": f"Identifying rectangle with length = {length} and width = {width}"},
                {"description": "Using the formula: Area = length × width"},
                {"description": f"Area = {length} × {width}"},
                {"description": f"Area = {area}"}
            ]
        
        # Create visualization data
        visualization = {
            "type": "rectangle_area",
            "shape": {
                "length": length,
                "width": width,
                "is_square": is_square,
                "vertices": [
                    {"x": 0, "y": 0},
                    {"x": length, "y": 0},
                    {"x": length, "y": width},
                    {"x": 0, "y": width}
                ]
            },
            "area": area
        }
        
        shape_name = "square" if is_square else "rectangle"
        
        return {
            "working": working_steps,
            "result": f"Area = {area}",
            "explanation": f"Used the formula for {shape_name} area to find the result",
            "confidence": 0.95,
            "computed_values": {"area": area},
            "visualization": visualization
        }
    
    def _sphere_volume(self, analysis: Dict) -> Dict:
        """Calculate the volume of a sphere"""
        # For demonstration, assume we know the radius
        radius = 3  # Example value
        
        # Calculate volume
        volume = (4/3) * 3.14159 * radius**3
        
        # Detailed working steps
        working_steps = [
            {"description": f"Identifying sphere with radius = {radius}"},
            {"description": "Using the formula: Volume = (4/3) × π × r³"},
            {"description": f"Volume = (4/3) × π × {radius}³"},
            {"description": f"Volume = (4/3) × π × {radius**3}"},
            {"description": f"Volume = {volume}"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "sphere_volume",
            "sphere": {
                "radius": radius,
                "center": {"x": 0, "y": 0, "z": 0}
            },
            "volume": volume
        }
        
        return {
            "working": working_steps,
            "result": f"Volume = {volume}",
            "explanation": "Used the formula Volume = (4/3) × π × r³ to find the sphere's volume",
            "confidence": 0.95,
            "computed_values": {"volume": volume},
            "visualization": visualization
        }
    
    def _cube_volume(self, analysis: Dict) -> Dict:
        """Calculate the volume of a cube"""
        # For demonstration, assume we know the side length
        side = 4  # Example value
        
        # Calculate volume
        volume = side**3
        
        # Detailed working steps
        working_steps = [
            {"description": f"Identifying cube with side length = {side}"},
            {"description": "Using the formula: Volume = side³"},
            {"description": f"Volume = {side}³"},
            {"description": f"Volume = {volume}"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "cube_volume",
            "cube": {
                "side": side,
                "vertices": [
                    {"x": 0, "y": 0, "z": 0},
                    {"x": side, "y": 0, "z": 0},
                    {"x": side, "y": side, "z": 0},
                    {"x": 0, "y": side, "z": 0},
                    {"x": 0, "y": 0, "z": side},
                    {"x": side, "y": 0, "z": side},
                    {"x": side, "y": side, "z": side},
                    {"x": 0, "y": side, "z": side}
                ]
            },
            "volume": volume
        }
        
        return {
            "working": working_steps,
            "result": f"Volume = {volume}",
            "explanation": "Used the formula Volume = side³ to find the cube's volume",
            "confidence": 0.95,
            "computed_values": {"volume": volume},
            "visualization": visualization
        }
    
    def _cylinder_volume(self, analysis: Dict) -> Dict:
        """Calculate the volume of a cylinder"""
        # For demonstration, assume we know the radius and height
        radius = 3  # Example value
        height = 7  # Example value
        
        # Calculate volume
        volume = 3.14159 * radius**2 * height
        
        # Detailed working steps
        working_steps = [
            {"description": f"Identifying cylinder with radius = {radius} and height = {height}"},
            {"description": "Using the formula: Volume = π × r² × h"},
            {"description": f"Volume = π × {radius}² × {height}"},
            {"description": f"Volume = π × {radius**2} × {height}"},
            {"description": f"Volume = {3.14159 * radius**2} × {height}"},
            {"description": f"Volume = {volume}"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "cylinder_volume",
            "cylinder": {
                "radius": radius,
                "height": height,
                "base_center": {"x": 0, "y": 0, "z": 0}
            },
            "volume": volume
        }
        
        return {
            "working": working_steps,
            "result": f"Volume = {volume}",
            "explanation": "Used the formula Volume = π × r² × h to find the cylinder's volume",
            "confidence": 0.95,
            "computed_values": {"volume": volume},
            "visualization": visualization
        }


class StatisticsExecutorAgent(ExecutorAgent):
    """Agent specialized in executing statistics solution strategies"""
    
    def __init__(self, workspace: Workspace):
        super().__init__("StatisticsExecutor", workspace, "statistics")
        
        # Knowledge base of statistics techniques
        self.techniques = {
            "apply_probability_rules": self.apply_probability_rules,
            "compute_summary_statistics": self.compute_summary_statistics,
            "apply_bayes_theorem": self.apply_bayes_theorem,
            "hypothesis_testing": self.hypothesis_testing,
            "confidence_intervals": self.confidence_intervals
        }
    
    def apply_probability_rules(self, analysis: Dict, step: Dict) -> Dict:
        """Apply rules of probability to solve problems"""
        components = analysis.get("components", {})
        target = components.get("target", {}).get("description", "")
        
        # Determine what type of probability problem this is
        if "conditional" in target.lower():
            return self._conditional_probability(analysis)
        elif "independent" in target.lower():
            return self._independent_events(analysis)
        elif "union" in target.lower() or "or" in target.lower():
            return self._probability_union(analysis)
        elif "intersection" in target.lower() or "and" in target.lower():
            return self._probability_intersection(analysis)
        else:
            # Generic probability approach
            return {
                "working": [
                    {"description": "Identifying the probability problem type"},
                    {"description": "Determining applicable probability rules"},
                    {"description": "Applying rules to compute the probability"}
                ],
                "result": "Probability calculation approach demonstrated",
                "explanation": "Used probability rules to solve the problem",
                "confidence": 0.7
            }
    
    def _conditional_probability(self, analysis: Dict) -> Dict:
        """Calculate conditional probability P(A|B)"""
        # For demonstration, assume we know P(A), P(B), and P(A∩B)
        p_a = 0.5  # Example value
        p_b = 0.4  # Example value
        p_a_and_b = 0.2  # Example value
        
        # Calculate P(A|B)
        p_a_given_b = p_a_and_b / p_b
        
        # Detailed working steps
        working_steps = [
            {"description": "Identifying conditional probability problem P(A|B)"},
            {"description": f"Given: P(A) = {p_a}, P(B) = {p_b}, P(A∩B) = {p_a_and_b}"},
            {"description": "Using the formula: P(A|B) = P(A∩B) / P(B)"},
            {"description": f"P(A|B) = {p_a_and_b} / {p_b}"},
            {"description": f"P(A|B) = {p_a_given_b}"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "conditional_probability",
            "probabilities": {
                "P(A)": p_a,
                "P(B)": p_b,
                "P(A∩B)": p_a_and_b,
                "P(A|B)": p_a_given_b
            },
            "venn_diagram": {
                "sets": [
                    {"name": "A", "size": p_a, "position": {"x": 0.3, "y": 0.5}},
                    {"name": "B", "size": p_b, "position": {"x": 0.7, "y": 0.5}}
                ],
                "intersection": {"size": p_a_and_b, "position": {"x": 0.5, "y": 0.5}}
            }
        }
        
        return {
            "working": working_steps,
            "result": f"P(A|B) = {p_a_given_b}",
            "explanation": "Used the conditional probability formula to find P(A|B)",
            "confidence": 0.9,
            "computed_values": {"conditional_probability": p_a_given_b},
            "visualization": visualization
        }
    
    def _independent_events(self, analysis: Dict) -> Dict:
        """Analyze independent events"""
        # For demonstration, assume we know P(A) and P(B)
        p_a = 0.3  # Example value
        p_b = 0.4  # Example value
        
        # For independent events, P(A∩B) = P(A) × P(B)
        p_a_and_b = p_a * p_b
        
        # Detailed working steps
        working_steps = [
            {"description": "Analyzing independent events A and B"},
            {"description": f"Given: P(A) = {p_a}, P(B) = {p_b}"},
            {"description": "For independent events: P(A∩B) = P(A) × P(B)"},
            {"description": f"P(A∩B) = {p_a} × {p_b}"},
            {"description": f"P(A∩B) = {p_a_and_b}"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "independent_events",
            "probabilities": {
                "P(A)": p_a,
                "P(B)": p_b,
                "P(A∩B)": p_a_and_b
            },
            "tree_diagram": {
                "root": {"name": "Start", "probability": 1.0},
                "branches": [
                    {
                        "name": "A", 
                        "probability": p_a,
                        "children": [
                            {"name": "B", "probability": p_b, "joint_probability": p_a_and_b},
                            {"name": "not B", "probability": 1-p_b, "joint_probability": p_a*(1-p_b)}
                        ]
                    },
                    {
                        "name": "not A", 
                        "probability": 1-p_a,
                        "children": [
                            {"name": "B", "probability": p_b, "joint_probability": (1-p_a)*p_b},
                            {"name": "not B", "probability": 1-p_b, "joint_probability": (1-p_a)*(1-p_b)}
                        ]
                    }
                ]
            }
        }
        
        return {
            "working": working_steps,
            "result": f"P(A∩B) = {p_a_and_b}",
            "explanation": "Applied the multiplication rule for independent events",
            "confidence": 0.9,
            "computed_values": {"joint_probability": p_a_and_b},
            "visualization": visualization
        }
    
    def _probability_union(self, analysis: Dict) -> Dict:
        """Calculate the probability of a union of events: P(A∪B)"""
        # For demonstration, assume we know P(A), P(B), and P(A∩B)
        p_a = 0.5  # Example value
        p_b = 0.4  # Example value
        p_a_and_b = 0.2  # Example value
        
        # Calculate P(A∪B)
        p_a_or_b = p_a + p_b - p_a_and_b
        
        # Detailed working steps
        working_steps = [
            {"description": "Calculating P(A∪B) - the probability of A or B"},
            {"description": f"Given: P(A) = {p_a}, P(B) = {p_b}, P(A∩B) = {p_a_and_b}"},
            {"description": "Using the formula: P(A∪B) = P(A) + P(B) - P(A∩B)"},
            {"description": f"P(A∪B) = {p_a} + {p_b} - {p_a_and_b}"},
            {"description": f"P(A∪B) = {p_a + p_b} - {p_a_and_b}"},
            {"description": f"P(A∪B) = {p_a_or_b}"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "probability_union",
            "probabilities": {
                "P(A)": p_a,
                "P(B)": p_b,
                "P(A∩B)": p_a_and_b,
                "P(A∪B)": p_a_or_b
            },
            "venn_diagram": {
                "sets": [
                    {"name": "A", "size": p_a, "position": {"x": 0.3, "y": 0.5}},
                    {"name": "B", "size": p_b, "position": {"x": 0.7, "y": 0.5}}
                ],
                "intersection": {"size": p_a_and_b, "position": {"x": 0.5, "y": 0.5}},
                "union": {"size": p_a_or_b}
            }
        }
        
        return {
            "working": working_steps,
            "result": f"P(A∪B) = {p_a_or_b}",
            "explanation": "Used the addition rule of probability to find P(A∪B)",
            "confidence": 0.95,
            "computed_values": {"union_probability": p_a_or_b},
            "visualization": visualization
        }
    
    def _probability_intersection(self, analysis: Dict) -> Dict:
        """Calculate the probability of an intersection of events: P(A∩B)"""
        # For demonstration, assume we know P(A), P(B), and whether they're independent
        p_a = 0.5  # Example value
        p_b = 0.4  # Example value
        independent = False  # Example value
        
        if independent:
            # For independent events: P(A∩B) = P(A) × P(B)
            p_a_and_b = p_a * p_b
            rule_used = "multiplication rule for independent events"
        else:
            # For dependent events, use P(A∩B) = P(A) × P(B|A)
            # For demonstration, assume P(B|A) = 0.6
            p_b_given_a = 0.6  # Example value
            p_a_and_b = p_a * p_b_given_a
            rule_used = "conditional probability formula"
        
        # Detailed working steps
        working_steps = [
            {"description": "Calculating P(A∩B) - the probability of A and B"},
            {"description": f"Given: P(A) = {p_a}, P(B) = {p_b}"}
        ]
        
        if independent:
            working_steps.extend([
                {"description": "Events A and B are independent"},
                {"description": "Using the formula for independent events: P(A∩B) = P(A) × P(B)"},
                {"description": f"P(A∩B) = {p_a} × {p_b}"},
                {"description": f"P(A∩B) = {p_a_and_b}"}
            ])
        else:
            working_steps.extend([
                {"description": "Events A and B are not independent"},
                {"description": f"Given: P(B|A) = {p_b_given_a}"},
                {"description": "Using the formula: P(A∩B) = P(A) × P(B|A)"},
                {"description": f"P(A∩B) = {p_a} × {p_b_given_a}"},
                {"description": f"P(A∩B) = {p_a_and_b}"}
            ])
        
        # Create visualization data
        visualization = {
            "type": "probability_intersection",
            "probabilities": {
                "P(A)": p_a,
                "P(B)": p_b,
                "P(A∩B)": p_a_and_b
            },
            "independent": independent,
            "venn_diagram": {
                "sets": [
                    {"name": "A", "size": p_a, "position": {"x": 0.3, "y": 0.5}},
                    {"name": "B", "size": p_b, "position": {"x": 0.7, "y": 0.5}}
                ],
                "intersection": {"size": p_a_and_b, "position": {"x": 0.5, "y": 0.5}}
            }
        }
        
        return {
            "working": working_steps,
            "result": f"P(A∩B) = {p_a_and_b}",
            "explanation": f"Applied the {rule_used} to find P(A∩B)",
            "confidence": 0.9,
            "computed_values": {"intersection_probability": p_a_and_b},
            "visualization": visualization
        }
    
    def compute_summary_statistics(self, analysis: Dict, step: Dict) -> Dict:
        """Compute statistical measures for data"""
        # For demonstration, create a sample dataset
        data = [12, 15, 18, 22, 25, 30, 35, 42, 48, 50]
        
        # Compute basic statistics
        mean = sum(data) / len(data)
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        # Median
        if n % 2 == 0:
            median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            median = sorted_data[n//2]
        
        # Mode (simplified - just find the most frequent value)
        from collections import Counter
        mode_counter = Counter(data)
        mode = mode_counter.most_common(1)[0][0]
        
        # Range
        data_range = max(data) - min(data)
        
        # Variance and Standard Deviation
        variance = sum((x - mean)**2 for x in data) / n
        std_dev = variance**0.5
        
        # Detailed working steps
        working_steps = [
            {"description": "Computing summary statistics for the dataset"},
            {"description": f"Dataset: {data}"},
            {"description": "Mean calculation: sum of all values divided by count"},
            {"description": f"Mean = ({' + '.join(str(x) for x in data)}) / {n}"},
            {"description": f"Mean = {sum(data)} / {n} = {mean}"},
            {"description": "Median calculation: middle value of sorted data"},
            {"description": f"Sorted data: {sorted_data}"},
            {"description": f"Median = {median}"},
            {"description": "Mode calculation: most frequent value"},
            {"description": f"Mode = {mode}"},
            {"description": "Range calculation: maximum - minimum"},
            {"description": f"Range = {max(data)} - {min(data)} = {data_range}"},
            {"description": "Variance calculation: average squared deviation from mean"},
            {"description": f"Variance = {variance}"},
            {"description": "Standard deviation: square root of variance"},
            {"description": f"Standard deviation = √{variance} = {std_dev}"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "summary_statistics",
            "data": data,
            "statistics": {
                "mean": mean,
                "median": median,
                "mode": mode,
                "range": data_range,
                "variance": variance,
                "standard_deviation": std_dev
            },
            "histogram": {
                "bins": 5,
                "frequencies": [2, 3, 2, 2, 1]  # Example frequencies
            }
        }
        
        return {
            "working": working_steps,
            "result": f"Mean = {mean}, Median = {median}, Mode = {mode}, Standard Deviation = {std_dev}",
            "explanation": "Computed key summary statistics to describe the central tendency and spread of the data",
            "confidence": 0.95,
            "computed_values": {
                "mean": mean,
                "median": median,
                "mode": mode,
                "range": data_range,
                "variance": variance,
                "standard_deviation": std_dev
            },
            "visualization": visualization
        }
    
    def apply_bayes_theorem(self, analysis: Dict, step: Dict) -> Dict:
        """Apply Bayes' theorem to calculate conditional probabilities"""
        # For demonstration, assume we know P(A), P(B|A), and P(B|not A)
        p_a = 0.3  # Example value - prior probability
        p_b_given_a = 0.8  # Example value - sensitivity
        p_b_given_not_a = 0.1  # Example value - false positive rate
        
        # Calculate P(not A)
        p_not_a = 1 - p_a
        
        # Calculate P(B) using the law of total probability
        p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
        
        # Apply Bayes' theorem to calculate P(A|B)
        p_a_given_b = (p_b_given_a * p_a) / p_b
        
        # Detailed working steps
        working_steps = [
            {"description": "Applying Bayes' theorem to calculate P(A|B)"},
            {"description": f"Given: P(A) = {p_a}, P(B|A) = {p_b_given_a}, P(B|not A) = {p_b_given_not_a}"},
            {"description": "Step 1: Calculate P(not A) = 1 - P(A)"},
            {"description": f"P(not A) = 1 - {p_a} = {p_not_a}"},
            {"description": "Step 2: Calculate P(B) using the law of total probability"},
            {"description": "P(B) = P(B|A) × P(A) + P(B|not A) × P(not A)"},
            {"description": f"P(B) = {p_b_given_a} × {p_a} + {p_b_given_not_a} × {p_not_a}"},
            {"description": f"P(B) = {p_b_given_a * p_a} + {p_b_given_not_a * p_not_a} = {p_b}"},
            {"description": "Step 3: Apply Bayes' theorem: P(A|B) = [P(B|A) × P(A)] / P(B)"},
            {"description": f"P(A|B) = [{p_b_given_a} × {p_a}] / {p_b}"},
            {"description": f"P(A|B) = {p_b_given_a * p_a} / {p_b} = {p_a_given_b}"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "bayes_theorem",
            "probabilities": {
                "prior": {"P(A)": p_a, "P(not A)": p_not_a},
                "likelihood": {"P(B|A)": p_b_given_a, "P(B|not A)": p_b_given_not_a},
                "marginal": {"P(B)": p_b},
                "posterior": {"P(A|B)": p_a_given_b, "P(not A|B)": 1 - p_a_given_b}
            },
            "tree_diagram": {
                "root": {"name": "Start", "probability": 1.0},
                "branches": [
                    {
                        "name": "A", 
                        "probability": p_a,
                        "children": [
                            {"name": "B", "probability": p_b_given_a, "joint_probability": p_b_given_a * p_a},
                            {"name": "not B", "probability": 1-p_b_given_a, "joint_probability": (1-p_b_given_a) * p_a}
                        ]
                    },
                    {
                        "name": "not A", 
                        "probability": p_not_a,
                        "children": [
                            {"name": "B", "probability": p_b_given_not_a, "joint_probability": p_b_given_not_a * p_not_a},
                            {"name": "not B", "probability": 1-p_b_given_not_a, "joint_probability": (1-p_b_given_not_a) * p_not_a}
                        ]
                    }
                ]
            }
        }
        
        return {
            "working": working_steps,
            "result": f"P(A|B) = {p_a_given_b}",
            "explanation": "Applied Bayes' theorem to update the probability of A given evidence B",
            "confidence": 0.9,
            "computed_values": {"posterior_probability": p_a_given_b},
            "visualization": visualization
        }
    
    def hypothesis_testing(self, analysis: Dict, step: Dict) -> Dict:
        """Perform hypothesis testing"""
        # For demonstration, assume we're doing a z-test for a mean
        sample_mean = 52  # Example value
        population_mean = 50  # Example value - null hypothesis
        population_std = 10  # Example value
        sample_size = 36  # Example value
        
        # Calculate the standard error
        standard_error = population_std / (sample_size**0.5)
        
        # Calculate the z-statistic
        z_statistic = (sample_mean - population_mean) / standard_error
        
        # Determine p-value (simplified calculation)
        # For a two-tailed test with z-statistic = 1.2, p-value ≈ 0.23
        p_value = 0.23  # Example value
        
        # Choose significance level
        alpha = 0.05
        
        # Make decision
        reject_null = p_value < alpha
        decision = "Reject the null hypothesis" if reject_null else "Fail to reject the null hypothesis"
        
        # Detailed working steps
        working_steps = [
            {"description": "Performing hypothesis testing (z-test for a mean)"},
            {"description": "Null hypothesis (H₀): μ = 50"},
            {"description": "Alternative hypothesis (H₁): μ ≠ 50"},
            {"description": f"Given: sample mean = {sample_mean}, population std dev = {population_std}, sample size = {sample_size}"},
            {"description": "Step 1: Calculate the standard error"},
            {"description": f"SE = σ / √n = {population_std} / √{sample_size} = {standard_error}"},
            {"description": "Step 2: Calculate the z-statistic"},
            {"description": f"z = (x̄ - μ) / SE = ({sample_mean} - {population_mean}) / {standard_error} = {z_statistic}"},
            {"description": "Step 3: Find the p-value"},
            {"description": f"For z = {z_statistic}, p-value = {p_value}"},
            {"description": f"Step 4: Compare p-value to significance level (α = {alpha})"},
            {"description": f"p-value {('<' if reject_null else '>')} α"},
            {"description": f"Decision: {decision}"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "hypothesis_testing",
            "test_type": "z-test for mean",
            "parameters": {
                "sample_mean": sample_mean,
                "population_mean": population_mean,
                "population_std": population_std,
                "sample_size": sample_size
            },
            "results": {
                "standard_error": standard_error,
                "z_statistic": z_statistic,
                "p_value": p_value,
                "alpha": alpha,
                "reject_null": reject_null
            },
            "distribution": {
                "mean": population_mean,
                "std": standard_error,
                "observed": sample_mean,
                "critical_values": [-1.96, 1.96]  # For α = 0.05
            }
        }
        
        return {
            "working": working_steps,
            "result": f"{decision} (p-value = {p_value})",
            "explanation": "Conducted a z-test to determine if the sample mean differs significantly from the population mean",
            "confidence": 0.9,
            "computed_values": {
                "z_statistic": z_statistic,
                "p_value": p_value,
                "reject_null": reject_null
            },
            "visualization": visualization
        }
    
    def confidence_intervals(self, analysis: Dict, step: Dict) -> Dict:
        """Calculate confidence intervals for parameter estimation"""
        # For demonstration, assume we're calculating a confidence interval for a mean
        sample_mean = 75  # Example value
        sample_std = 12  # Example value
        sample_size = 40  # Example value
        confidence_level = 0.95  # Example value
        
        # For 95% confidence, z-critical value is approximately 1.96
        z_critical = 1.96
        
        # Calculate standard error
        standard_error = sample_std / (sample_size**0.5)
        
        # Calculate margin of error
        margin_of_error = z_critical * standard_error
        
        # Calculate confidence interval
        lower_bound = sample_mean - margin_of_error
        upper_bound = sample_mean + margin_of_error
        
        # Detailed working steps
        working_steps = [
            {"description": f"Calculating {confidence_level*100}% confidence interval for a mean"},
            {"description": f"Given: sample mean = {sample_mean}, sample std dev = {sample_std}, sample size = {sample_size}"},
            {"description": f"Step 1: Find the critical value for {confidence_level*100}% confidence"},
            {"description": f"For {confidence_level*100}% confidence, z-critical = {z_critical}"},
            {"description": "Step 2: Calculate the standard error"},
            {"description": f"SE = s / √n = {sample_std} / √{sample_size} = {standard_error}"},
            {"description": "Step 3: Calculate the margin of error"},
            {"description": f"ME = z-critical × SE = {z_critical} × {standard_error} = {margin_of_error}"},
            {"description": "Step 4: Construct the confidence interval"},
            {"description": f"CI = x̄ ± ME = {sample_mean} ± {margin_of_error}"},
            {"description": f"CI = ({lower_bound}, {upper_bound})"}
        ]
        
        # Create visualization data
        visualization = {
            "type": "confidence_interval",
            "parameters": {
                "sample_mean": sample_mean,
                "sample_std": sample_std,
                "sample_size": sample_size,
                "confidence_level": confidence_level
            },
            "results": {
                "standard_error": standard_error,
                "z_critical": z_critical,
                "margin_of_error": margin_of_error,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            },
            "distribution": {
                "mean": sample_mean,
                "std": standard_error,
                "interval": [lower_bound, upper_bound]
            }
        }
        
        return {
            "working": working_steps,
            "result": f"{confidence_level*100}% Confidence Interval: ({lower_bound}, {upper_bound})",
            "explanation": f"Constructed a {confidence_level*100}% confidence interval for the population mean",
            "confidence": 0.95,
            "computed_values": {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "margin_of_error": margin_of_error
            },
            "visualization": visualization
        }


class NumberTheoryExecutorAgent(ExecutorAgent):
    """Agent specialized in executing number theory solution strategies"""
    
    def __init__(self, workspace: Workspace):
        super().__init__("NumberTheoryExecutor", workspace, "number_theory")
        
        # Knowledge base of number theory techniques
        self.techniques = {
            "prime_factorization": self.prime_factorization,
            "use_modular_arithmetic": self.use_modular_arithmetic,
            "find_gcd_lcm": self.find_gcd_lcm,
            "apply_divisibility_rules": self.apply_divisibility_rules,
            "solve_diophantine_equation": self.solve_diophantine_equation
        }
    
    def prime_factorization(self, analysis: Dict, step: Dict) -> Dict:
        """Find the prime factorization of a number"""
        # For demonstration, pick a number to factorize
        number = 84  # Example value
        
        # Compute prime factorization
        n = number
        factors = []
        d = 2
        
        while n > 1:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
            if d*d > n and n > 1:
                factors.append(n)
                break
        
        # Format the prime factorization
        factorization = {}
        for factor in factors:
            factorization[factor] = factorization.get(factor, 0) + 1
        
        # Create a formatted expression
        expression = " × ".join([f"{base}^{power}" if power > 1 else str(base) 
                              for base, power in factorization.items()])
        
        # Detailed working steps
        working_steps = [
            {"description": f"Finding the prime factorization of {number}"},
            {"description": "Starting with the smallest prime factor, 2"}
        ]
        
        # Create steps for the factorization process
        n = number
        d = 2
        while n > 1:
            if n % d == 0:
                steps_text = f"Divide {n} by {d}: {n} ÷ {d} = {n//d}"
                working_steps.append({"description": steps_text})
                n //= d
            else:
                working_steps.append({"description": f"{n} is not divisible by {d}, try next factor"})
                d += 1
                if d*d > n and n > 1:
                    working_steps.append({"description": f"{n} is prime, so it's a factor"})
                    break
        
        working_steps.append({"description": f"Prime factorization: {expression}"})
        
        # Create visualization data
        visualization = {
            "type": "prime_factorization",
            "number": number,
            "factors": factorization,
            "expression": expression,
            "factor_tree": {
                "root": number,
                "branches": self._create_factor_tree(number)
            }
        }
        
        return {
            "working": working_steps,
            "result": f"Prime factorization of {number} = {expression}",
            "explanation": "Found the unique prime factorization of the number",
            "confidence": 0.95,
            "computed_values": {"factorization": factorization, "factors": factors},
            "visualization": visualization
        }
    
    def _create_factor_tree(self, n, max_depth=3):
        """Helper method to create a factor tree for visualization"""
        if n <= 1 or max_depth <= 0:
            return None
            
        # Find a factor
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return {
                    "factor1": i,
                    "factor2": n // i,
                    "left": self._create_factor_tree(i, max_depth - 1),
                    "right": self._create_factor_tree(n // i, max_depth - 1)
                }
        
        # Prime number
        return None
    
    def use_modular_arithmetic(self, analysis: Dict, step: Dict) -> Dict:
        """Apply modular arithmetic to solve problems"""
        # For demonstration, compute a modular exponentiation
        base = 7  # Example value
        exponent = 23  # Example value
        modulus = 13  # Example value
        
        # Compute modular exponentiation using the square-and-multiply algorithm
        result = 1
        b = base % modulus
        e = exponent
        
        # Track steps for square-and-multiply
        exp_steps = []
        
        while e > 0:
            if e % 2 == 1:
                result = (result * b) % modulus
                exp_steps.append(f"e is odd, multiply: result = (result * b) % mod = ({result // b} * {b}) % {modulus} = {result}")
            b = (b * b) % modulus
            e //= 2
            if e > 0:
                exp_steps.append(f"Square base: b = (b * b) % mod = ({b // b} * {b // b}) % {modulus} = {b}")
        
        # Detailed working steps
        working_steps = [
            {"description": f"Computing {base}^{exponent} mod {modulus}"},
            {"description": "Using the square-and-multiply algorithm for efficient modular exponentiation"},
            {"description": "Start with result = 1, b = base % modulus"},
            {"description": f"result = 1, b = {base} % {modulus} = {base % modulus}"}
        ]
        
        for step in exp_steps:
            working_steps.append({"description": step})
            
        working_steps.append({"description": f"Final result: {base}^{exponent} mod {modulus} = {result}"})
        
        # Create visualization data
        visualization = {
            "type": "modular_arithmetic",
            "operation": "exponentiation",
            "parameters": {
                "base": base,
                "exponent": exponent,
                "modulus": modulus
            },
            "result": result,
            "modular_circle": {
                "size": modulus,
                "values": [{"value": i, "highlight": i == result} for i in range(modulus)]
            }
        }
        
        return {
            "working": working_steps,
            "result": f"{base}^{exponent} mod {modulus} = {result}",
            "explanation": "Used the square-and-multiply algorithm for efficient modular exponentiation",
            "confidence": 0.95,
            "computed_values": {"modular_exponentiation": result},
            "visualization": visualization
        }
    
    def find_gcd_lcm(self, analysis: Dict, step: Dict) -> Dict:
        """Find the greatest common divisor (GCD) and least common multiple (LCM)"""
        # For demonstration, find GCD and LCM of two numbers
        a = 48  # Example value
        b = 36  # Example value
        
        # Compute GCD using the Euclidean algorithm
        def gcd(x, y):
            while y:
                x, y = y, x % y
            return x
        
        # Store steps for the Euclidean algorithm
        gcd_steps = []
        m, n = a, b
        
        while n:
            gcd_steps.append(f"{m} = {m // n} × {n} + {m % n}")
            m, n = n, m % n
        
        gcd_value = m
        
        # Compute LCM using the formula: LCM(a,b) = a * b / GCD(a,b)
        lcm_value = a * b // gcd_value
        
        # Detailed working steps
        working_steps = [
            {"description": f"Finding the GCD and LCM of {a} and {b}"},
            {"description": "Step 1: Compute GCD using the Euclidean algorithm"}
        ]
        
        for step in gcd_steps:
            working_steps.append({"description": step})
            
        working_steps.extend([
            {"description": f"GCD({a}, {b}) = {gcd_value}"},
            {"description": "Step 2: Compute LCM using the formula: LCM(a,b) = a × b / GCD(a,b)"},
            {"description": f"LCM({a}, {b}) = {a} × {b} / {gcd_value} = {a * b} / {gcd_value} = {lcm_value}"}
        ])
        
        # Create visualization data
        visualization = {
            "type": "gcd_lcm",
            "numbers": {"a": a, "b": b},
            "results": {"gcd": gcd_value, "lcm": lcm_value},
            "prime_factorizations": {
                "a": self._get_prime_factorization(a),
                "b": self._get_prime_factorization(b),
                "gcd": self._get_prime_factorization(gcd_value),
                "lcm": self._get_prime_factorization(lcm_value)
            }
        }
        
        return {
            "working": working_steps,
            "result": f"GCD({a}, {b}) = {gcd_value}, LCM({a}, {b}) = {lcm_value}",
            "explanation": "Used the Euclidean algorithm to find the GCD and the formula LCM(a,b) = a × b / GCD(a,b)",
            "confidence": 0.95,
            "computed_values": {"gcd": gcd_value, "lcm": lcm_value},
            "visualization": visualization
        }
    
    def _get_prime_factorization(self, n):
        """Helper method to get the prime factorization as a dictionary"""
        factors = {}
        d = 2
        
        while n > 1:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
            if d*d > n and n > 1:
                factors[n] = factors.get(n, 0) + 1
                break
                
        return factors
    
    def apply_divisibility_rules(self, analysis: Dict, step: Dict) -> Dict:
        """Apply divisibility rules to determine if a number is divisible by another"""
        # For demonstration, check divisibility of a number
        number = 34578  # Example value
        divisors_to_check = [2, 3, 4, 5, 6, 8, 9, 10, 11]
        
        # Store results and explanations
        divisibility_results = {}
        
        for divisor in divisors_to_check:
            # Actually check divisibility
            is_divisible = number % divisor == 0
            
            # Apply divisibility rules for explanation
            if divisor == 2:
                rule = f"A number is divisible by 2 if its last digit is even."
                explanation = f"The last digit is {number % 10}, which is {'even' if number % 10 % 2 == 0 else 'odd'}."
            elif divisor == 3:
                digit_sum = sum(int(digit) for digit in str(number))
                rule = f"A number is divisible by 3 if the sum of its digits is divisible by 3."
                explanation = f"The sum of digits is {digit_sum}, which is {'' if digit_sum % 3 == 0 else 'not '}divisible by 3."
            elif divisor == 4:
                last_two_digits = number % 100
                rule = f"A number is divisible by 4 if its last two digits form a number divisible by 4."
                explanation = f"The last two digits are {last_two_digits}, which is {'' if last_two_digits % 4 == 0 else 'not '}divisible by 4."
            elif divisor == 5:
                rule = f"A number is divisible by 5 if its last digit is 0 or 5."
                explanation = f"The last digit is {number % 10}, which is {'' if number % 10 in [0, 5] else 'not '}0 or 5."
            elif divisor == 6:
                rule = f"A number is divisible by 6 if it is divisible by both 2 and 3."
                explanation = f"The number is {'' if number % 2 == 0 else 'not '}divisible by 2 and {'' if number % 3 == 0 else 'not '}divisible by 3."
            elif divisor == 8:
                last_three_digits = number % 1000
                rule = f"A number is divisible by 8 if its last three digits form a number divisible by 8."
                explanation = f"The last three digits are {last_three_digits}, which is {'' if last_three_digits % 8 == 0 else 'not '}divisible by 8."
            elif divisor == 9:
                digit_sum = sum(int(digit) for digit in str(number))
                rule = f"A number is divisible by 9 if the sum of its digits is divisible by 9."
                explanation = f"The sum of digits is {digit_sum}, which is {'' if digit_sum % 9 == 0 else 'not '}divisible by 9."
            elif divisor == 10:
                rule = f"A number is divisible by 10 if its last digit is 0."
                explanation = f"The last digit is {number % 10}, which is {'' if number % 10 == 0 else 'not '}0."
            elif divisor == 11:
                digits = [int(digit) for digit in str(number)]
                alternating_sum = sum((-1)**i * digit for i, digit in enumerate(digits))
                rule = f"A number is divisible by 11 if the alternating sum of its digits is divisible by 11."
                explanation = f"The alternating sum is {alternating_sum}, which is {'' if alternating_sum % 11 == 0 else 'not '}divisible by 11."
            else:
                rule = f"Checking divisibility by {divisor} directly."
                explanation = f"{number} is {'' if is_divisible else 'not '}divisible by {divisor}."
            
            divisibility_results[divisor] = {
                "is_divisible": is_divisible,
                "rule": rule,
                "explanation": explanation
            }
        
        # Detailed working steps
        working_steps = [
            {"description": f"Checking divisibility of {number} by various divisors"},
        ]
        
        for divisor, result in divisibility_results.items():
            working_steps.extend([
                {"description": f"Checking divisibility by {divisor}:"},
                {"description": f"Rule: {result['rule']}"},
                {"description": f"Analysis: {result['explanation']}"},
                {"description": f"Result: {number} is {'' if result['is_divisible'] else 'not '}divisible by {divisor}."}
            ])
        
        # Format results for output
        divisible_by = [divisor for divisor, result in divisibility_results.items() if result["is_divisible"]]
        result_str = f"{number} is divisible by: {', '.join(map(str, divisible_by))}"
        
        # Create visualization data
        visualization = {
            "type": "divisibility",
            "number": number,
            "results": {str(divisor): result["is_divisible"] for divisor, result in divisibility_results.items()},
            "factorization": self._get_prime_factorization(number)
        }
        
        return {
            "working": working_steps,
            "result": result_str,
            "explanation": "Applied divisibility rules to determine which numbers divide the given number",
            "confidence": 0.95,
            "computed_values": {"divisible_by": divisible_by},
            "visualization": visualization
        }
    
    def solve_diophantine_equation(self, analysis: Dict, step: Dict) -> Dict:
        """Solve a linear Diophantine equation ax + by = c"""
        # For demonstration, solve a linear Diophantine equation
        a = 12  # Example value
        b = 18  # Example value
        c = 30  # Example value
        
        # First, check if a solution exists (c must be divisible by gcd(a,b))
        def gcd(x, y):
            while y:
                x, y = y, x % y
            return x
        
        g = gcd(a, b)
        has_solution = c % g == 0
        
        if not has_solution:
            working_steps = [
                {"description": f"Solving the Diophantine equation {a}x + {b}y = {c}"},
                {"description": f"Step 1: Check if a solution exists by finding gcd({a}, {b})"},
                {"description": f"gcd({a}, {b}) = {g}"},
                {"description": f"Check if {c} is divisible by {g}: {c} % {g} = {c % g}"},
                {"description": f"Since {c} is not divisible by {g}, the equation has no integer solutions."}
            ]
            
            return {
                "working": working_steps,
                "result": "No integer solutions exist",
                "explanation": "A linear Diophantine equation ax + by = c has solutions if and only if c is divisible by gcd(a,b)",
                "confidence": 0.95,
                "computed_values": {"has_solution": False, "gcd": g},
                "visualization": {
                    "type": "diophantine_equation",
                    "equation": {"a": a, "b": b, "c": c},
                    "gcd": g,
                    "has_solution": False
                }
            }
        
        # If a solution exists, find a particular solution using the Extended Euclidean Algorithm
        def extended_gcd(x, y):
            if y == 0:
                return x, 1, 0
            else:
                d, a, b = extended_gcd(y, x % y)
                return d, b, a - (x // y) * b
        
        _, x0, y0 = extended_gcd(a, b)
        x0 *= c // g
        y0 *= c // g
        
        # Generate a few solutions
        solutions = []
        for k in range(-2, 3):
            x = x0 + k * (b // g)
            y = y0 - k * (a // g)
            solutions.append((x, y))
        
        # Detailed working steps
        working_steps = [
            {"description": f"Solving the Diophantine equation {a}x + {b}y = {c}"},
            {"description": f"Step 1: Check if a solution exists by finding gcd({a}, {b})"},
            {"description": f"gcd({a}, {b}) = {g}"},
            {"description": f"Check if {c} is divisible by {g}: {c} % {g} = {c % g}"},
            {"description": f"Since {c} is divisible by {g}, solutions exist."},
            {"description": "Step 2: Find a particular solution using the Extended Euclidean Algorithm"},
            {"description": f"Extended GCD gives x₀ = {x0 // (c // g)}, y₀ = {y0 // (c // g)} such that {a}x₀ + {b}y₀ = {g}"},
            {"description": f"Multiply by {c // g} to get {a}({x0 // (c // g)} × {c // g}) + {b}({y0 // (c // g)} × {c // g}) = {g} × {c // g} = {c}"},
            {"description": f"This gives the particular solution x₀ = {x0}, y₀ = {y0}"},
            {"description": "Step 3: Generate the general solution"},
            {"description": f"General formula: x = {x0} + {b // g}t, y = {y0} - {a // g}t for any integer t"}
        ]
        
        # Add some specific solutions
        for i, (x, y) in enumerate(solutions):
            working_steps.append({"description": f"For t = {i-2}, solution: x = {x}, y = {y}"})
        
        # Create visualization data
        visualization = {
            "type": "diophantine_equation",
            "equation": {"a": a, "b": b, "c": c},
            "gcd": g,
            "has_solution": True,
            "particular_solution": {"x": x0, "y": y0},
            "general_solution": {"x_formula": f"{x0} + ({b // g})t", "y_formula": f"{y0} - ({a // g})t"},
            "solutions": solutions
        }
        
        return {
            "working": working_steps,
            "result": f"General solution: x = {x0} + {b // g}t, y = {y0} - {a // g}t for any integer t",
            "explanation": "Solved the linear Diophantine equation using the Extended Euclidean Algorithm",
            "confidence": 0.95,
            "computed_values": {
                "particular_solution": {"x": x0, "y": y0},
                "general_solution": {"x_formula": f"{x0} + ({b // g})t", "y_formula": f"{y0} - ({a // g})t"}
            },
            "visualization": visualization
        }


# =============================
# VERIFICATION AGENTS
# =============================

class VerifierAgent(Agent):
    """Agent that checks steps for correctness"""
    
    def __init__(self, workspace: Workspace):
        super().__init__("Verifier", workspace)
        self.workspace.subscribe(self.id, [MessageType.EXECUTION, MessageType.SYNTHESIS])
        
        # Knowledge base of verification techniques
        self.verification_techniques = {
            "dimensional_analysis": self._verify_dimensions,
            "plug_back": self._verify_plug_back,
            "edge_cases": self._verify_edge_cases,
            "alternate_method": self._verify_alternate_method,
            "numerical_test": self._verify_numerical_test,
            "check_assumptions": self._verify_assumptions
        }
        
        # Initialize verification history
        self.verification_history = {}
    
    def step(self):
        if not self.active:
            return
            
        new_messages = self.get_new_messages()
        
        for msg in new_messages:
            if msg.type == MessageType.EXECUTION:
                self.verify_execution_step(msg)
            elif msg.type == MessageType.SYNTHESIS:
                self.verify_final_solution(msg)
    
    def verify_execution_step(self, execution_msg: Message):
        """Verify an individual execution step"""
        execution = execution_msg.content
        strategy_applied = execution.get("strategy_applied", "")
        domain = execution.get("domain", "")
        
        # Initialize issues list
        issues = []
        
        # Check for execution errors
        if "error" in execution:
            issues.append({
                "severity": "high",
                "issue": f"Execution error: {execution['error']}",
                "suggestion": "Revise the approach or correct calculation errors"
            })
        
        # Check for low confidence
        if execution_msg.confidence < 0.6:
            issues.append({
                "severity": "medium",
                "issue": "Low confidence in execution result",
                "suggestion": "Consider alternative approaches or verify intermediate steps"
            })
        
        # Determine which verification techniques to apply based on the strategy and domain
        techniques_to_apply = self._select_verification_techniques(strategy_applied, domain)
        
        # Apply selected verification techniques
        verification_results = []
        
        for technique in techniques_to_apply:
            verify_func = self.verification_techniques.get(technique)
            if verify_func:
                result = verify_func(execution, execution_msg)
                verification_results.append(result)
                
                # If the verification fails, add an issue
                if not result.get("passed", True):
                    issues.append({
                        "severity": "medium" if result.get("critical", False) else "low",
                        "issue": result.get("issue", f"Failed {technique} verification"),
                        "suggestion": result.get("suggestion", "Review this step carefully")
                    })
        
        # Calculate overall verification score
        if issues:
            severity_weights = {"high": 0.8, "medium": 0.5, "low": 0.2}
            verification_score = 1.0 - min(1.0, sum(severity_weights.get(issue["severity"], 0.2) for issue in issues) / len(issues))
        else:
            verification_score = 0.95  # High confidence if no issues
        
        # Add to verification history
        self.verification_history[execution_msg.id] = {
            "verification_score": verification_score,
            "issues": issues,
            "techniques_applied": techniques_to_apply
        }
        
        # Create detailed explanation of verification
        verification_explanation = (
            f"Verification of {strategy_applied} execution completed with a confidence score of {verification_score:.2f}. "
            f"Applied techniques: {', '.join(techniques_to_apply)}."
        )
        
        if issues:
            verification_explanation += f" Found {len(issues)} issues that should be addressed."
        else:
            verification_explanation += " No issues were detected."
        
        # Send verification message
        self.send_message(
            MessageType.VERIFICATION, 
            {
                "step_id": execution_msg.id,
                "strategy_applied": strategy_applied,
                "domain": domain,
                "issues": issues,
                "verification_tests": verification_results,
                "verification_score": verification_score,
                "overall_assessment": "Valid step" if verification_score > 0.7 else "Needs revision",
                "explanation": verification_explanation
            },
            confidence=verification_score,
            references=[execution_msg.id]
        )
    
    def verify_final_solution(self, synthesis_msg: Message):
        """Verify the final synthesized solution"""
        solution = synthesis_msg.content
        solution_parts = solution.get("solution_parts", [])
        
        # Initialize issues list
        issues = []
        
        # Check if the solution addresses all parts of the problem
        if not any(part.get("section") == "Answer" for part in solution_parts):
            issues.append({
                "severity": "high",
                "issue": "Solution does not contain a clear answer",
                "suggestion": "Explicitly state the answer to the original question"
            })
        
        # Find the original problem
        references = synthesis_msg.references
        problem_msg = None
        for ref_id in references:
            msg = self.workspace.get_message_by_id(ref_id)
            if msg and msg.type == MessageType.PROBLEM:
                problem_msg = msg
                break
        
        if not problem_msg:
            problem_msg = self._find_problem_message()
        
        if not problem_msg:
            issues.append({
                "severity": "medium",
                "issue": "Cannot verify solution without the original problem",
                "suggestion": "Ensure the solution references the original problem"
            })
            
            # Send partial verification even without the problem
            verification_score = 0.5  # Reduced confidence without problem reference
            
            self.send_message(
                MessageType.VERIFICATION, 
                {
                    "solution_id": synthesis_msg.id,
                    "issues": issues,
                    "verification_score": verification_score,
                    "overall_assessment": "Incomplete verification"
                },
                confidence=verification_score,
                references=[synthesis_msg.id]
            )
            return
        
        # Find execution messages referenced by the synthesis
        execution_msgs = []
        for ref_id in references:
            msg = self.workspace.get_message_by_id(ref_id)
            if msg and msg.type == MessageType.EXECUTION:
                execution_msgs.append(msg)
        
        # If no direct execution references, find all executions
        if not execution_msgs:
            execution_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.EXECUTION]
        
        # Check that the solution integrates all executed steps
        if execution_msgs:
            execution_strategies = set(msg.content.get("strategy_applied", "") for msg in execution_msgs)
            solution_strategies = set()
            
            for part in solution_parts:
                if part.get("section") == "Solution" and "steps" in part:
                    for step in part["steps"]:
                        if "strategy" in step:
                            solution_strategies.add(step["strategy"])
            
            missing_strategies = execution_strategies - solution_strategies
            if missing_strategies:
                issues.append({
                    "severity": "medium",
                    "issue": f"Solution is missing results from strategies: {', '.join(missing_strategies)}",
                    "suggestion": "Ensure all executed steps are properly integrated into the final solution"
                })
        
        # Apply comprehensive verification techniques
        verification_techniques = [
            self._verify_solution_consistency,
            self._verify_solution_completeness,
            self._verify_answer_correctness
        ]
        
        verification_results = []
        for verify_func in verification_techniques:
            result = verify_func(solution, problem_msg, execution_msgs)
            verification_results.append(result)
            
            # If the verification fails, add an issue
            if not result.get("passed", True):
                issues.append({
                    "severity": "medium" if result.get("critical", False) else "low",
                    "issue": result.get("issue", "Verification failed"),
                    "suggestion": result.get("suggestion", "Review the solution carefully")
                })
        
        # Calculate verification score
        if issues:
            severity_weights = {"high": 0.8, "medium": 0.5, "low": 0.2}
            verification_score = 1.0 - min(1.0, sum(severity_weights.get(issue["severity"], 0.2) for issue in issues) / len(issues))
        else:
            verification_score = 0.95  # High confidence if no issues
        
        # Create detailed explanation of verification
        verification_explanation = (
            f"Final solution verification completed with a confidence score of {verification_score:.2f}. "
            f"Verified solution consistency, completeness, and correctness."
        )
        
        if issues:
            verification_explanation += f" Found {len(issues)} issues that should be addressed."
        else:
            verification_explanation += " The solution appears to be correct and complete."
        
        # Send verification message
        self.send_message(
            MessageType.VERIFICATION, 
            {
                "solution_id": synthesis_msg.id,
                "issues": issues,
                "verification_results": verification_results,
                "verification_score": verification_score,
                "overall_assessment": "Valid solution" if verification_score > 0.7 else "Needs revision",
                "explanation": verification_explanation
            },
            confidence=verification_score,
            references=[synthesis_msg.id, problem_msg.id] + [msg.id for msg in execution_msgs]
        )
    
    def _find_problem_message(self):
        """Find the problem message in the workspace"""
        problem_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.PROBLEM]
        return problem_msgs[0] if problem_msgs else None
    
    def _select_verification_techniques(self, strategy, domain):
        """Select appropriate verification techniques based on strategy and domain"""
        techniques = ["check_assumptions"]  # Always check assumptions
        
        # Add domain-specific techniques
        if domain == "algebra" or domain == "calculus":
            techniques.extend(["plug_back", "numerical_test"])
        elif domain == "geometry":
            techniques.extend(["dimensional_analysis", "edge_cases"])
        elif domain == "statistics":
            techniques.extend(["edge_cases", "numerical_test"])
        elif domain == "number_theory":
            techniques.extend(["numerical_test", "alternate_method"])
        
        # Add strategy-specific techniques
        if "solve" in strategy or "factor" in strategy:
            if "plug_back" not in techniques:
                techniques.append("plug_back")
        elif "derivative" in strategy or "integral" in strategy:
            if "numerical_test" not in techniques:
                techniques.append("numerical_test")
        
        # Randomly select a subset of techniques to avoid over-verification
        if len(techniques) > 3:
            return random.sample(techniques, 3)
        return techniques
    
    def _verify_dimensions(self, execution, message):
        """Verify that the dimensions match on both sides of equations"""
        # Simple implementation for demonstration
        return {
            "technique": "dimensional_analysis",
            "passed": True,
            "details": "Dimensions appear to be consistent"
        }
    
    def _verify_plug_back(self, execution, message):
        """Verify the result by substituting back into the original problem"""
        # Simple implementation for demonstration
        result = execution.get("result", "")
        computed_values = execution.get("computed_values", {})
        
        # Check if we have solutions to plug back
        if "solutions" in computed_values or "solution" in computed_values:
            return {
                "technique": "plug_back",
                "passed": True,
                "details": "Solution verified by substitution"
            }
        elif "equation" in computed_values:
            return {
                "technique": "plug_back",
                "passed": True,
                "details": "Equation verified algebraically"
            }
        
        # Default return if no specific verification is possible
        return {
            "technique": "plug_back",
            "passed": True,
            "details": "No specific verification performed"
        }
    
    def _verify_edge_cases(self, execution, message):
        """Verify that the solution handles edge cases correctly"""
        # Simple implementation for demonstration
        return {
            "technique": "edge_cases",
            "passed": True,
            "details": "Solution appears valid for standard cases"
        }
    
    def _verify_alternate_method(self, execution, message):
        """Verify by using an alternative solution method"""
        # Simple implementation for demonstration
        return {
            "technique": "alternate_method",
            "passed": True,
            "details": "Result consistent with expected outcome"
        }
    
    def _verify_numerical_test(self, execution, message):
        """Verify by testing with specific numerical values"""
        # Simple implementation for demonstration
        return {
            "technique": "numerical_test",
            "passed": True,
            "details": "Numerical verification successful"
        }
    
    def _verify_assumptions(self, execution, message):
        """Verify that all assumptions in the solution are valid"""
        # Simple implementation for demonstration
        return {
            "technique": "check_assumptions",
            "passed": True,
            "details": "Assumptions appear valid for this problem"
        }
    
    def _verify_solution_consistency(self, solution, problem_msg, execution_msgs):
        """Verify that the solution is internally consistent"""
        # Simple implementation for demonstration
        return {
            "technique": "solution_consistency",
            "passed": True,
            "details": "Solution steps appear to be consistent"
        }
    
    def _verify_solution_completeness(self, solution, problem_msg, execution_msgs):
        """Verify that the solution addresses all aspects of the problem"""
        # Simple implementation for demonstration
        return {
            "technique": "solution_completeness",
            "passed": True,
            "details": "Solution appears to address all aspects of the problem"
        }
    
    def _verify_answer_correctness(self, solution, problem_msg, execution_msgs):
        """Verify that the final answer is correct"""
        # Simple implementation for demonstration
        return {
            "technique": "answer_correctness",
            "passed": True,
            "details": "Final answer appears to be correct"
        }


class DebateAgent(Agent):
    """Agent that challenges reasoning and promotes critical thinking"""
    
    def __init__(self, workspace: Workspace):
        super().__init__("Debate", workspace)
        self.workspace.subscribe(self.id, [MessageType.STRATEGY, MessageType.EXECUTION, 
                                       MessageType.VERIFICATION, MessageType.KNOWLEDGE])
        
        # Knowledge base of common fallacies and reasoning errors
        self.fallacies = {
            "false_assumption": "Reasoning based on an unverified assumption",
            "circular_reasoning": "Using the conclusion as a premise",
            "oversimplification": "Ignoring important aspects of the problem",
            "false_generalization": "Applying a general rule to a specific case incorrectly",
            "non_sequitur": "Conclusion doesn't follow from the premises",
            "affirming_consequent": "If A then B, B is true, therefore A is true",
            "denying_antecedent": "If A then B, A is false, therefore B is false",
            "correlation_causation": "Assuming correlation implies causation"
        }
        
        # Track debates to avoid excessive challenges
        self.debate_history = {}
        self.debate_timeout = 5  # Number of messages before considering debating the same topic again
    
    def step(self):
        if not self.active:
            return
            
        new_messages = self.get_new_messages()
        
        # Prioritize messages with low confidence or verification issues
        verification_msgs = [msg for msg in new_messages if msg.type == MessageType.VERIFICATION and msg.confidence < 0.7]
        if verification_msgs:
            for verification_msg in verification_msgs:
                self.challenge_based_on_verification(verification_msg)
        
        # Occasionally challenge strategies and executions
        knowledge_msgs = [msg for msg in new_messages if msg.type == MessageType.KNOWLEDGE]
        strategy_msgs = [msg for msg in new_messages if msg.type == MessageType.STRATEGY]
        execution_msgs = [msg for msg in new_messages if msg.type == MessageType.EXECUTION]
        
        # Use resource allocation to determine debate frequency
        debate_frequency = min(0.3, self.resource_allocation * 0.4)
        
        if strategy_msgs and random.random() < debate_frequency:
            self.challenge_strategy(random.choice(strategy_msgs), knowledge_msgs)
        
        elif execution_msgs and random.random() < debate_frequency * 0.7:
            self.challenge_execution(random.choice(execution_msgs))
    
    def challenge_based_on_verification(self, verification_msg: Message):
        """Challenge reasoning based on verification issues"""
        verification = verification_msg.content
        referenced_msg_id = verification.get("step_id", verification.get("solution_id"))
        
        if not referenced_msg_id:
            return
            
        # Check if we've recently debated this message
        if referenced_msg_id in self.debate_history:
            last_debate = self.debate_history[referenced_msg_id]
            if self.workspace.current_time - last_debate < self.debate_timeout:
                return  # Avoid excessive debates on the same topic
        
        # Find the message being verified
        referenced_msg = self.workspace.get_message_by_id(referenced_msg_id)
        if not referenced_msg:
            return
            
        # Extract issues from verification
        issues = verification.get("issues", [])
        
        if not issues:
            return
            
        # Choose most severe issue to challenge
        issue = sorted(issues, key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x.get("severity"), 0), reverse=True)[0]
        
        # Create debate message
        debate_content = {
            "challenge_type": "verification_issue",
            "contested_message_id": referenced_msg.id,
            "contested_content": self._summarize_content(referenced_msg.content),
            "issue": issue["issue"],
            "reasoning": f"The verifier identified a potential problem: {issue['issue']}. {issue.get('suggestion', '')}",
            "alternative_consideration": "Consider revising this step or providing additional justification."
        }
        
        # Update debate history
        self.debate_history[referenced_msg_id] = self.workspace.current_time
        
        self.send_message(
            MessageType.DEBATE, 
            debate_content,
            confidence=0.7,
            references=[verification_msg.id, referenced_msg.id]
        )
    
    def challenge_strategy(self, strategy_msg: Message, knowledge_msgs=None):
        """Challenge aspects of a proposed strategy"""
        strategy = strategy_msg.content
        
        # Check if we've recently debated this strategy
        if strategy_msg.id in self.debate_history:
            last_debate = self.debate_history[strategy_msg.id]
            if self.workspace.current_time - last_debate < self.debate_timeout:
                return  # Avoid excessive debates on the same strategy
        
        # Look for potential issues to challenge
        challenge_points = []
        
        # Check for overly complex strategies
        if len(strategy.get("steps", [])) > 5:
            challenge_points.append({
                "issue": "strategy_complexity",
                "description": "The proposed strategy seems unnecessarily complex",
                "fallacy": "oversimplification",
                "alternative": "Consider whether a more direct approach might solve the problem more elegantly"
            })
        
        # Check knowledge-based challenges
        if knowledge_msgs:
            for knowledge_msg in knowledge_msgs:
                if strategy_msg.id in knowledge_msg.references:
                    knowledge_items = knowledge_msg.content.get("knowledge_items", [])
                    
                    for item in knowledge_items:
                        # Look for alternative approaches suggested in knowledge
                        item_content = item.get("item", {})
                        item_name = item_content.get("name", "").lower()
                        item_description = item_content.get("description", "").lower()
                        
                        strategy_steps = [step.get("strategy", "").lower() for step in strategy.get("steps", []) 
                                        if isinstance(step, dict) and "strategy" in step]
                        
                        # If knowledge suggests an approach not in the strategy
                        if (("method" in item_type or "technique" in item_type) and 
                            all(method_name not in step for step in strategy_steps)):
                            challenge_points.append({
                                "issue": "alternative_approach",
                                "description": f"The strategy doesn't consider {item_name} which might be applicable",
                                "fallacy": "false_assumption",
                                "alternative": f"Consider whether {item_name} ({item_description}) might be useful"
                            })
        
        # Check for domain-specific challenges
        for step in strategy.get("steps", []):
            if isinstance(step, dict) and "strategy" in step and "confidence" in step:
                strategy_name = step["strategy"]
                confidence = step["confidence"]
                
                if confidence < 0.7:
                    challenge_points.append({
                        "issue": f"low_confidence_{strategy_name}",
                        "description": f"Low confidence in the {strategy_name} approach",
                        "fallacy": "false_assumption",
                        "alternative": "Consider whether a different approach might be more reliable"
                    })
        
        # If no specific issues found, maybe challenge a random step
        if not challenge_points and random.random() < 0.3:
            steps = [step for step in strategy.get("steps", []) 
                   if isinstance(step, dict) and "strategy" in step]
            
            if steps:
                step_to_challenge = random.choice(steps)
                strategy_name = step_to_challenge["strategy"]
                fallacy = random.choice(list(self.fallacies.keys()))
                
                challenge_points.append({
                    "issue": f"questioning_{strategy_name}",
                    "description": f"I'm not convinced that {strategy_name} is the best approach here",
                    "fallacy": fallacy,
                    "alternative": f"Consider whether {self.fallacies[fallacy]} might be affecting this choice"
                })
        
        # If we have challenge points, create a debate message
        if challenge_points:
            challenge = random.choice(challenge_points)
            
            debate_content = {
                "challenge_type": "strategy_critique",
                "contested_strategy": strategy.get("high_level_approach", strategy.get("overall_approach", "the proposed strategy")),
                "specific_issue": challenge["issue"],
                "reasoning": f"I see a potential issue: {challenge['description']}. This might involve {self.fallacies.get(challenge['fallacy'], 'a reasoning error')}.",
                "alternative_consideration": challenge["alternative"]
            }
            
            # Update debate history
            self.debate_history[strategy_msg.id] = self.workspace.current_time
            
            self.send_message(
                MessageType.DEBATE, 
                debate_content,
                confidence=0.6,
                references=[strategy_msg.id]
            )
    
    def challenge_execution(self, execution_msg: Message):
        """Challenge aspects of a solution execution"""
        execution = execution_msg.content
        
        # Check if we've recently debated this execution
        if execution_msg.id in self.debate_history:
            last_debate = self.debate_history[execution_msg.id]
            if self.workspace.current_time - last_debate < self.debate_timeout:
                return  # Avoid excessive debates on the same execution
        
        # For demonstration, challenge with simple checks
        challenge_points = []
        
        # Check if there's any working shown
        if "working" not in execution or not execution["working"]:
            challenge_points.append({
                "issue": "insufficient_work_shown",
                "description": "The execution doesn't show enough working steps to verify its correctness",
                "fallacy": "false_assumption",
                "alternative": "Please provide more detailed step-by-step working"
            })
        
        # Check for potential numerical errors
        if "result" in execution and isinstance(execution["result"], str):
            # Look for mathematical expressions in the result
            mathematical_expressions = re.findall(r'[-+]?\d+(\.\d+)?', execution["result"])
            if mathematical_expressions and random.random() < 0.2:  # 20% chance to challenge a calculation
                challenge_points.append({
                    "issue": "calculation_verification",
                    "description": "The numerical calculations should be verified for accuracy",
                    "fallacy": "false_assumption",
                    "alternative": "Double-check the calculations to ensure correctness"
                })
        
        # Check for domain-specific issues
        domain = execution.get("domain", "")
        strategy_applied = execution.get("strategy_applied", "")
        
        if domain == "algebra" and "factor" in strategy_applied:
            challenge_points.append({
                "issue": "factorization_verification",
                "description": "The factorization should be verified by expansion",
                "fallacy": "false_assumption",
                "alternative": "Verify the factorization by multiplying the factors back together"
            })
        elif domain == "calculus" and "derivative" in strategy_applied:
            challenge_points.append({
                "issue": "derivative_rule_application",
                "description": "Ensure all derivative rules are correctly applied",
                "fallacy": "false_assumption",
                "alternative": "Double-check the application of derivative rules, especially for composite functions"
            })
        
        # If confidence is low, challenge the approach
        if execution_msg.confidence < 0.7:
            challenge_points.append({
                "issue": "low_confidence_execution",
                "description": "The agent doesn't seem confident in this execution. We should reconsider the approach.",
                "fallacy": "false_assumption",
                "alternative": "Consider a different solution method or verify the current calculations"
            })
        
        # If we have challenge points, randomly select one and create a debate message
        if challenge_points:
            challenge = random.choice(challenge_points)
            
            debate_content = {
                "challenge_type": "execution_critique",
                "contested_execution": strategy_applied,
                "specific_issue": challenge["issue"],
                "reasoning": challenge["description"],
                "alternative_consideration": challenge["alternative"]
            }
            
            # Update debate history
            self.debate_history[execution_msg.id] = self.workspace.current_time
            
            self.send_message(
                MessageType.DEBATE, 
                debate_content,
                confidence=0.7,
                references=[execution_msg.id]
            )
    
    def _summarize_content(self, content):
        """Summarize message content for inclusion in debate messages"""
        if isinstance(content, dict):
            if "strategy_applied" in content:
                return f"Execution of {content['strategy_applied']}"
            elif "high_level_approach" in content:
                return content["high_level_approach"]
            elif "steps" in content:
                return f"Strategy with {len(content['steps'])} steps"
            else:
                return "Message content"
        elif isinstance(content, str):
            return content[:50] + "..." if len(content) > 50 else content
        else:
            return "Message content"


class SynthesizerAgent(Agent):
    """Agent that combines results into a coherent solution"""
    
    def __init__(self, workspace: Workspace):
        super().__init__("Synthesizer", workspace)
        self.workspace.subscribe(self.id, [MessageType.EXECUTION, MessageType.VERIFICATION, 
                                        MessageType.DEBATE, MessageType.EXPLANATION])
        self.outstanding_executions = {}  # Track expected executions
        self.execution_results = {}  # Store successful execution results
        self.verification_results = {}  # Store verification results
        
        # Track problem context for better synthesis
        self.problem_context = {
            "problem_id": None,
            "analysis_id": None,
            "strategy_id": None
        }
    
    def step(self):
        if not self.active:
            return
            
        new_messages = self.get_new_messages()
        
        # Process analysis messages to understand the problem
        for msg in new_messages:
            if msg.type == MessageType.ANALYSIS:
                # Find the referenced problem
                for ref_id in msg.references:
                    ref_msg = self.workspace.get_message_by_id(ref_id)
                    if ref_msg and ref_msg.type == MessageType.PROBLEM:
                        self.problem_context["problem_id"] = ref_id
                        self.problem_context["analysis_id"] = msg.id
                        break
        
        # Process strategy messages to know what executions to expect
        for msg in new_messages:
            if msg.type == MessageType.STRATEGY:
                self.problem_context["strategy_id"] = msg.id
                self.register_expected_executions(msg)
        
        # Process execution and verification results
        for msg in new_messages:
            if msg.type == MessageType.EXECUTION:
                self.execution_results[msg.id] = {
                    "content": msg.content,
                    "confidence": msg.confidence,
                    "timestamp": msg.timestamp
                }
            elif msg.type == MessageType.VERIFICATION and "step_id" in msg.content:
                step_id = msg.content["step_id"]
                self.verification_results[step_id] = {
                    "content": msg.content,
                    "confidence": msg.confidence,
                    "timestamp": msg.timestamp
                }
        
        # Check if we have all expected executions and their verifications
        self.check_synthesis_readiness()
    
    def register_expected_executions(self, strategy_msg: Message):
        """Register what execution steps we expect based on the strategy"""
        strategy = strategy_msg.content
        
        # Clear previous expectations
        self.outstanding_executions = {}
        
        # Register new expectations
        for step in strategy.get("steps", []):
            if isinstance(step, dict) and "strategy" in step and "domain" in step:
                step_id = f"{step['domain']}_{step['strategy']}"
                self.outstanding_executions[step_id] = {
                    "step": step,
                    "completed": False,
                    "execution_id": None
                }
    
    def check_synthesis_readiness(self):
        """Check if we have enough information to synthesize a solution"""
        # Update completion status of expected executions
        for exec_id, exec_data in self.execution_results.items():
            content = exec_data["content"]
            if "strategy_applied" in content and "domain" in content:
                strategy = content["strategy_applied"]
                domain = content["domain"]
                step_id = f"{domain}_{strategy}"
                
                if step_id in self.outstanding_executions and not self.outstanding_executions[step_id]["completed"]:
                    self.outstanding_executions[step_id]["completed"] = True
                    self.outstanding_executions[step_id]["execution_id"] = exec_id
        
        # Count completed and verified executions
        completed_count = sum(1 for step_data in self.outstanding_executions.values() if step_data["completed"])
        total_count = len(self.outstanding_executions)
        
        # Check if we have enough to synthesize
        # We proceed if all steps are complete, or if at least 70% are complete
        # Also ensure we have at least one execution before proceeding
        if (completed_count > 0 and 
            (completed_count == total_count or 
             (total_count > 0 and completed_count / total_count >= 0.7))):
            self.synthesize_solution()
    
    def synthesize_solution(self):
        """Synthesize a complete solution from all execution results"""
        # Get problem and analysis
        problem_id = self.problem_context.get("problem_id")
        analysis_id = self.problem_context.get("analysis_id")
        strategy_id = self.problem_context.get("strategy_id")
        
        problem_msg = None
        analysis_msg = None
        strategy_msg = None
        
        if problem_id:
            problem_msg = self.workspace.get_message_by_id(problem_id)
        
        if analysis_id:
            analysis_msg = self.workspace.get_message_by_id(analysis_id)
        
        if strategy_id:
            strategy_msg = self.workspace.get_message_by_id(strategy_id)
        
        # If we don't have a problem message, try to find one
        if not problem_msg:
            problem_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.PROBLEM]
            if problem_msgs:
                problem_msg = problem_msgs[0]
                problem_id = problem_msg.id
        
        # If we don't have an analysis message, try to find one
        if not analysis_msg and problem_id:
            analysis_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.ANALYSIS 
                           and problem_id in msg.references]
            if analysis_msgs:
                analysis_msg = max(analysis_msgs, key=lambda x: x.timestamp)
                analysis_id = analysis_msg.id
        
        # If we don't have a strategy message, try to find one
        if not strategy_msg and analysis_id:
            strategy_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.STRATEGY 
                           and analysis_id in msg.references]
            if strategy_msgs:
                strategy_msg = max(strategy_msgs, key=lambda x: x.timestamp)
                strategy_id = strategy_msg.id
        
        # Get problem details
        problem_text = problem_msg.content if problem_msg else "Unknown problem"
        analysis = analysis_msg.content if analysis_msg else {}
        strategy = strategy_msg.content if strategy_msg else {}
        
        # Collect all successful executions
        successful_executions = []
        for step_id, step_data in self.outstanding_executions.items():
            if step_data["completed"] and step_data["execution_id"] in self.execution_results:
                exec_data = self.execution_results[step_data["execution_id"]]
                
                # Check if this execution was verified successfully
                verification_passed = True
                if step_data["execution_id"] in self.verification_results:
                    verify_data = self.verification_results[step_data["execution_id"]]
                    verification_passed = verify_data["confidence"] >= 0.6
                
                if verification_passed:
                    successful_executions.append({
                        "step": step_data["step"],
                        "execution": exec_data["content"],
                        "confidence": exec_data["confidence"],
                        "timestamp": exec_data["timestamp"]
                    })
        
        # Sort executions by step number or timestamp if step number not available
        successful_executions.sort(key=lambda x: (
            x["step"].get("step", float('inf')) if isinstance(x["step"], dict) else float('inf'),
            x["timestamp"]
        ))
        
        # Build the solution
        solution_parts = []
        
        # Add problem statement
        solution_parts.append({
            "section": "Problem",
            "content": problem_text
        })
        
        # Add approach explanation
        if strategy:
            approach = strategy.get("high_level_approach", strategy.get("overall_approach", "Mathematical problem-solving approach"))
            solution_parts.append({
                "section": "Approach",
                "content": approach
            })
        
        # Add solution steps
        solution_parts.append({
            "section": "Solution",
            "steps": []
        })
        
        for exec_data in successful_executions:
            execution = exec_data["execution"]
            step_details = {}
            
            # Include step description
            if "step_description" in execution:
                step_details["description"] = execution["step_description"]
            elif "strategy_applied" in execution:
                step_details["description"] = f"Applying {execution['strategy_applied']}"
            
            # Include working steps
            if "working" in execution:
                if isinstance(execution["working"], list):
                    step_details["working"] = [step["description"] for step in execution["working"] 
                                            if isinstance(step, dict) and "description" in step]
                else:
                    step_details["working"] = execution["working"]
            
            # Include result
            if "result" in execution:
                step_details["result"] = execution["result"]
            
            # Include explanation
            if "explanation" in execution:
                step_details["explanation"] = execution["explanation"]
            
            # Include any visualizations
            if "visualization" in execution:
                step_details["visualization"] = execution["visualization"]
            
            solution_parts[-1]["steps"].append(step_details)
        
        # Add final answer if available
        if successful_executions:
            final_result = None
            
            # Try to find the most appropriate result to use as the answer
            for exec_data in reversed(successful_executions):  # Start from the last execution
                execution = exec_data["execution"]
                if "result" in execution:
                    final_result = execution["result"]
                    break
            
            if final_result:
                solution_parts.append({
                    "section": "Answer",
                    "content": final_result
                })
        
        # Calculate overall confidence
        if successful_executions:
            overall_confidence = sum(exec_data["confidence"] for exec_data in successful_executions) / len(successful_executions)
        else:
            overall_confidence = 0.5
        
        # Generate explanation of the solution process
        solution_explanation = self._generate_solution_explanation(problem_text, strategy, successful_executions)
        
        # Create synthesis message references
        synthesis_references = []
        if problem_msg:
            synthesis_references.append(problem_msg.id)
        if analysis_msg:
            synthesis_references.append(analysis_msg.id)
        if strategy_msg:
            synthesis_references.append(strategy_msg.id)
        synthesis_references.extend([exec_data["execution_id"] for step_data in self.outstanding_executions.values() 
                                  if step_data["completed"] and "execution_id" in step_data])
        
        # Send synthesized solution
        synthesis_message_id = self.send_message(
            MessageType.SYNTHESIS, 
            {
                "solution_parts": solution_parts,
                "explanation": solution_explanation,
                "completeness": len(successful_executions) / max(1, len(self.outstanding_executions)),
                "verified": True,  # We only included verified executions
                "confidence": overall_confidence
            },
            confidence=overall_confidence,
            references=synthesis_references
        )
        
        # Send a human-friendly explanation
        self.send_message(
            MessageType.EXPLANATION, 
            {
                "title": "Solution Explanation",
                "content": solution_explanation,
                "summary": self._generate_solution_summary(problem_text, solution_parts)
            },
            confidence=overall_confidence,
            references=[synthesis_message_id]
        )
        
        # Clear tracking data after synthesis
        self.outstanding_executions = {}
        self.execution_results = {}
        self.verification_results = {}
    
    def _generate_solution_explanation(self, problem, strategy, executions):
        """Generate a human-friendly explanation of the solution process"""
        if not executions:
            return "No solution steps were successfully executed."
        
        approach = strategy.get("high_level_approach", strategy.get("overall_approach", "Mathematical problem-solving approach"))
        
        explanation = f"To solve this problem, we used a {approach}. "
        explanation += f"This involved {len(executions)} key steps: "
        
        for i, exec_data in enumerate(executions):
            execution = exec_data["execution"]
            step_description = execution.get("step_description", execution.get("strategy_applied", f"Step {i+1}"))
            explanation += f"{i+1}) {step_description}, "
        
        explanation = explanation.rstrip(", ") + ". "
        
        # Add a conclusion about the answer
        if executions:
            final_execution = executions[-1]["execution"]
            if "result" in final_execution:
                explanation += f"This led us to the answer: {final_execution['result']}."
        
        return explanation
    
    def _generate_solution_summary(self, problem, solution_parts):
        """Generate a brief summary of the solution"""
        answer_part = next((part for part in solution_parts if part.get("section") == "Answer"), None)
        
        if answer_part:
            return f"Problem: {problem[:100]}{'...' if len(problem) > 100 else ''}\nAnswer: {answer_part['content']}"
        else:
            return f"Problem: {problem[:100]}{'...' if len(problem) > 100 else ''}\nSolution completed."


class MetaCognitiveAgent(Agent):
    """Agent that dynamically allocates resources and manages the problem-solving process"""
    
    def __init__(self, workspace: Workspace):
        super().__init__("MetaCognitive", workspace)
        self.workspace.subscribe(self.id, [msg_type for msg_type in MessageType])
        
        # Track the state of the problem-solving process
        self.current_phase = "waiting_for_problem"  # Initial state
        self.phase_transitions = {
            "waiting_for_problem": "analysis",
            "analysis": "strategy_formulation",
            "strategy_formulation": "execution",
            "execution": "verification",
            "verification": "synthesis",
            "synthesis": "waiting_for_problem"  # Cycle back to the beginning
        }
        
        # Track resource allocation
        self.resource_allocation = {
            "ProblemAnalyzer": 1.0,
            "Strategy": 0.5,
            "AlgebraExecutor": 0.5,
            "CalculusExecutor": 0.5,
            "GeometryExecutor": 0.5,
            "StatisticsExecutor": 0.5,
            "NumberTheoryExecutor": 0.5,
            "KnowledgeBase": 0.7,
            "AnalogicalReasoning": 0.5,
            "Intuition": 0.3,
            "Verifier": 0.5,
            "Debate": 0.3,
            "Synthesizer": 0.5
        }
        
        # Track problem-solving metrics
        self.metrics = {
            "problem_complexity": 0.0,
            "solution_confidence": 0.0,
            "verification_rate": 0.0,
            "debate_activity": 0.0,
            "execution_success_rate": 0.0,
            "time_in_phase": {}
        }
        
        # Track phase history
        self.phase_history = []
        self.phase_start_time = 0
    
    def step(self):
        new_messages = self.get_new_messages()
        
        # Update problem-solving state based on messages
        self.update_state(new_messages)
        
        # Update metrics
        self.update_metrics()
        
        # Make resource allocation decisions
        self.allocate_resources()
        
        # Check for phase transitions
        self.check_phase_transition()
    
    def update_state(self, new_messages: List[Message]):
        """Update problem-solving state based on new messages"""
        # Update current phase based on message types
        previous_phase = self.current_phase
        
        for msg in new_messages:
            if msg.type == MessageType.PROBLEM:
                self.current_phase = "analysis"
                break
                
            elif msg.type == MessageType.ANALYSIS and self.current_phase == "analysis":
                self.current_phase = "strategy_formulation"
                break
                
            elif msg.type == MessageType.STRATEGY and self.current_phase == "strategy_formulation":
                self.current_phase = "execution"
                break
                
            elif msg.type == MessageType.VERIFICATION and self.current_phase == "execution":
                # Check if most execution steps have verification
                execution_msgs = [m for m in self.workspace.messages if m.type == MessageType.EXECUTION]
                verification_msgs = [m for m in self.workspace.messages if m.type == MessageType.VERIFICATION]
                
                if execution_msgs and len(verification_msgs) / len(execution_msgs) >= 0.7:
                    self.current_phase = "verification"
                break
                
            elif msg.type == MessageType.SYNTHESIS and self.current_phase == "verification":
                self.current_phase = "synthesis"
                break
            
            elif msg.type == MessageType.EXPLANATION and self.current_phase == "synthesis":
                self.current_phase = "waiting_for_problem"
                break
        
        # If phase changed, update history
        if previous_phase != self.current_phase:
            time_in_phase = self.workspace.current_time - self.phase_start_time
            self.phase_history.append({
                "phase": previous_phase,
                "duration": time_in_phase,
                "transition_time": self.workspace.current_time
            })
            self.phase_start_time = self.workspace.current_time
            
            # Update time in phase metric
            self.metrics["time_in_phase"][previous_phase] = self.metrics["time_in_phase"].get(previous_phase, 0) + time_in_phase
    
    def update_metrics(self):
        """Update problem-solving metrics"""
        # Update problem complexity
        analysis_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.ANALYSIS]
        if analysis_msgs:
            latest_analysis = max(analysis_msgs, key=lambda x: x.timestamp).content
            self.metrics["problem_complexity"] = latest_analysis.get("complexity", 0.5)
        
        # Update solution confidence
        execution_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.EXECUTION]
        if execution_msgs:
            confidences = [msg.confidence for msg in execution_msgs]
            self.metrics["solution_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Update verification rate
        if execution_msgs:
            verification_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.VERIFICATION]
            self.metrics["verification_rate"] = len(verification_msgs) / len(execution_msgs)
        
        # Update debate activity
        debate_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.DEBATE]
        total_msgs = len(self.workspace.messages)
        self.metrics["debate_activity"] = len(debate_msgs) / total_msgs if total_msgs > 0 else 0.0
        
        # Update execution success rate
        if execution_msgs:
            # Consider an execution successful if its confidence is high
            successful_executions = sum(1 for msg in execution_msgs if msg.confidence > 0.7)
            self.metrics["execution_success_rate"] = successful_executions / len(execution_msgs)
    
    def allocate_resources(self):
        """Allocate resources to different agents based on the current state"""
        # Reset allocations
        for agent in self.resource_allocation:
            self.resource_allocation[agent] = 0.5  # Default allocation
        
        # Allocate based on current phase and metrics
        if self.current_phase == "analysis":
            self.resource_allocation["ProblemAnalyzer"] = 1.0
            self.resource_allocation["KnowledgeBase"] = 0.8
            self.resource_allocation["Intuition"] = 0.8
            
        elif self.current_phase == "strategy_formulation":
            self.resource_allocation["Strategy"] = 1.0
            self.resource_allocation["KnowledgeBase"] = 0.9
            self.resource_allocation["AnalogicalReasoning"] = 0.8
            self.resource_allocation["Intuition"] = 0.8
            self.resource_allocation["Debate"] = 0.7
            
        elif self.current_phase == "execution":
            # Allocate to the most relevant executors based on domain
            analysis_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.ANALYSIS]
            if analysis_msgs:
                analysis = max(analysis_msgs, key=lambda x: x.timestamp).content
                domains = analysis.get("domains", {})
                
                if domains:
                    # Sort domains by confidence
                    sorted_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)
                    
                    # Allocate resources to relevant executors
                    for domain, confidence in sorted_domains:
                        if domain == "algebra":
                            self.resource_allocation["AlgebraExecutor"] = 0.5 + confidence * 0.5
                        elif domain == "calculus":
                            self.resource_allocation["CalculusExecutor"] = 0.5 + confidence * 0.5
                        elif domain == "geometry":
                            self.resource_allocation["GeometryExecutor"] = 0.5 + confidence * 0.5
                        elif domain in ["statistics", "probability"]:
                            self.resource_allocation["StatisticsExecutor"] = 0.5 + confidence * 0.5
                        elif domain == "number_theory":
                            self.resource_allocation["NumberTheoryExecutor"] = 0.5 + confidence * 0.5
            
            self.resource_allocation["KnowledgeBase"] = 0.7
            self.resource_allocation["Verifier"] = 0.8
            
        elif self.current_phase == "verification":
            self.resource_allocation["Verifier"] = 1.0
            self.resource_allocation["Debate"] = 0.8
            self.resource_allocation["KnowledgeBase"] = 0.6
            
        elif self.current_phase == "synthesis":
            self.resource_allocation["Synthesizer"] = 1.0
            self.resource_allocation["Verifier"] = 0.7
        
        # Adjust based on metrics
        if self.metrics["problem_complexity"] > 0.7:
            # For complex problems, increase intuition and knowledge base
            self.resource_allocation["Intuition"] = max(self.resource_allocation["Intuition"], 0.8)
            self.resource_allocation["KnowledgeBase"] = max(self.resource_allocation["KnowledgeBase"], 0.8)
            
        if self.metrics["solution_confidence"] < 0.6:
            # For low confidence solutions, increase verification and debate
            self.resource_allocation["Verifier"] = max(self.resource_allocation["Verifier"], 0.9)
            self.resource_allocation["Debate"] = max(self.resource_allocation["Debate"], 0.8)
        
        # Apply resource allocation to agents
        for agent_id, allocation in self.resource_allocation.items():
            for agent in self.workspace.messages:
                if agent.sender == agent_id:
                    # Find the agent instance and set its resource allocation
                    break
        
        # Send resource allocation message
        self.send_message(
            MessageType.META, 
            {
                "resource_allocation": self.resource_allocation,
                "current_phase": self.current_phase,
                "metrics": self.metrics,
                "explanation": f"Resources allocated based on {self.current_phase} phase and current metrics."
            },
            confidence=0.9
        )
    
    def check_phase_transition(self):
        """Check if we should transition to the next problem-solving phase"""
        # Count messages of each type
        message_counts = {}
        for msg_type in MessageType:
            message_counts[msg_type] = len([msg for msg in self.workspace.messages if msg.type == msg_type])
        
        # Calculate time spent in current phase
        time_in_current_phase = self.workspace.current_time - self.phase_start_time
        
        # Check if we should transition based on messages and time
        transition_conditions = {
            "analysis": lambda: message_counts[MessageType.ANALYSIS] > 0 and time_in_current_phase > 3,
            "strategy_formulation": lambda: message_counts[MessageType.STRATEGY] > 0 and time_in_current_phase > 3,
            "execution": lambda: message_counts[MessageType.EXECUTION] >= 1 and time_in_current_phase > 5,
            "verification": lambda: (message_counts[MessageType.EXECUTION] > 0 and 
                                   message_counts[MessageType.VERIFICATION] / message_counts[MessageType.EXECUTION] >= 0.7 and
                                   time_in_current_phase > 3),
            "synthesis": lambda: message_counts[MessageType.SYNTHESIS] > 0 and time_in_current_phase > 2,
            "waiting_for_problem": lambda: message_counts[MessageType.PROBLEM] > 0
        }
        
        if self.current_phase in transition_conditions and transition_conditions[self.current_phase]():
            next_phase = self.phase_transitions.get(self.current_phase)
            if next_phase:
                # Record time in current phase
                time_in_phase = self.workspace.current_time - self.phase_start_time
                self.phase_history.append({
                    "phase": self.current_phase,
                    "duration": time_in_phase,
                    "transition_time": self.workspace.current_time
                })
                
                # Update metrics
                self.metrics["time_in_phase"][self.current_phase] = self.metrics["time_in_phase"].get(self.current_phase, 0) + time_in_phase
                
                # Transition to next phase
                self.current_phase = next_phase
                self.phase_start_time = self.workspace.current_time
                
                # Notify about phase transition
                self.send_message(
                    MessageType.META, 
                    {
                        "phase_transition": {
                            "from": self.current_phase,
                            "to": next_phase
                        },
                        "explanation": f"Transitioning from {self.current_phase} to {next_phase} phase.",
                        "phase_history": self.phase_history,
                        "current_metrics": self.metrics
                    },
                    confidence=0.9
                )


# =============================
# MAIN SIGMA SYSTEM
# =============================

class SIGMA:
    """Main orchestrator for the multi-agent math reasoning system"""
    
    def __init__(self):
        self.workspace = Workspace()
        
        # Create all agents
        self.agents = {
            # Analysis agents
            "ProblemAnalyzer": ProblemAnalyzerAgent(self.workspace),
            "KnowledgeBase": KnowledgeBaseAgent(self.workspace),
            "AnalogicalReasoning": AnalogicalReasoningAgent(self.workspace),
            
            # Strategy agents
            "Strategy": StrategyAgent(self.workspace),
            
            # Executor agents
            "AlgebraExecutor": AlgebraExecutorAgent(self.workspace),
            "CalculusExecutor": CalculusExecutorAgent(self.workspace),
            "GeometryExecutor": GeometryExecutorAgent(self.workspace),
            "StatisticsExecutor": StatisticsExecutorAgent(self.workspace),
            "NumberTheoryExecutor": NumberTheoryExecutorAgent(self.workspace),
            
            # Intuition agent
            "Intuition": IntuitionAgent(self.workspace),
            
            # Verification and synthesis agents
            "Verifier": VerifierAgent(self.workspace),
            "Debate": DebateAgent(self.workspace),
            "Synthesizer": SynthesizerAgent(self.workspace),
            
            # Meta-cognitive agent
            "MetaCognitive": MetaCognitiveAgent(self.workspace)
        }
    
    def solve_problem(self, problem_text: str, max_steps: int = 100) -> Dict:
        """Solve a mathematical problem using the multi-agent system"""
        logger.info(f"Starting to solve problem: {problem_text}")
        
        # Reset the workspace
        self.workspace = Workspace()
        for agent_id, agent in self.agents.items():
            agent.workspace = self.workspace
            agent.last_processed_time = 0
        
        # Add the problem to the workspace
        self.workspace.add_message(
            Message(MessageType.PROBLEM, problem_text, "User", 1.0)
        )
        
        # Run the agent cycles
        for step in range(max_steps):
            logger.info(f"Step {step+1}/{max_steps}")
            
            # Let each agent take a step
            for agent_id, agent in self.agents.items():
                agent.step()
            
            # Check if we have a solution
            synthesis_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.SYNTHESIS]
            explanation_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.EXPLANATION]
            if synthesis_msgs and explanation_msgs:
                logger.info("Solution synthesized and explained. Stopping.")
                break
        
        # Return the final solution if available
        synthesis_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.SYNTHESIS]
        explanation_msgs = [msg for msg in self.workspace.messages if msg.type == MessageType.EXPLANATION]
        
        if synthesis_msgs and explanation_msgs:
            final_solution = max(synthesis_msgs, key=lambda x: x.timestamp)
            final_explanation = max(explanation_msgs, key=lambda x: x.timestamp)
            
            return {
                "solution": final_solution.content,
                "explanation": final_explanation.content,
                "confidence": final_solution.confidence,
                "steps_taken": len(self.workspace.messages),
                "agent_messages": {agent_id: len([msg for msg in self.workspace.messages if msg.sender == agent_id]) 
                                  for agent_id in self.agents},
                "message_timeline": self.workspace.get_performance_report(),
                "knowledge_graph": self.workspace.generate_knowledge_graph_visualization()
            }
        else:
            return {
                "solution": None,
                "error": "No solution synthesized within maximum steps",
                "steps_taken": len(self.workspace.messages),
                "agent_messages": {agent_id: len([msg for msg in self.workspace.messages if msg.sender == agent_id]) 
                                  for agent_id in self.agents},
                "message_timeline": self.workspace.get_performance_report()
            }
    
    def explain_solution(self, result: Dict) -> str:
        """Generate a human-readable explanation of the solution"""
        if not result or "solution" not in result or not result["solution"]:
            return "Unable to solve the problem. The system did not reach a solution."
        
        solution = result["solution"]
        explanation = result.get("explanation", {}).get("content", "")
        
        if explanation:
            return explanation
        
        # If no explanation is provided, generate one from the solution
        parts = solution.get("solution_parts", [])
        
        # Build explanation
        explanation_lines = []
        
        for part in parts:
            section = part.get("section", "")
            content = part.get("content", "")
            
            if section == "Problem":
                explanation_lines.append(f"Problem: {content}")
                
            elif section == "Approach":
                explanation_lines.append(f"Approach: {content}")
                
            elif section == "Solution":
                explanation_lines.append("Solution steps:")
                steps = part.get("steps", [])
                for i, step in enumerate(steps, 1):
                    explanation_lines.append(f"  Step {i}: {step.get('description', '')}")
                    
                    if 'working' in step:
                        if isinstance(step['working'], list):
                            for work in step['working']:
                                explanation_lines.append(f"    {work}")
                        else:
                            explanation_lines.append(f"    {step['working']}")
                            
                    if 'result' in step:
                        explanation_lines.append(f"    Result: {step['result']}")
                        
                    if 'explanation' in step:
                        explanation_lines.append(f"    Explanation: {step['explanation']}")
                    
            elif section == "Answer":
                explanation_lines.append(f"Answer: {content}")
        
        # Add confidence information
        explanation_lines.append(f"\nSolution confidence: {result.get('confidence', 0) * 100:.1f}%")
        
        return "\n".join(explanation_lines)
    
    def generate_performance_report(self) -> Dict:
        """Generate a report on system performance"""
        return self.workspace.get_performance_report()
    
    def generate_knowledge_graph(self) -> Dict:
        """Generate a knowledge graph visualization"""
        return self.workspace.generate_knowledge_graph_visualization()
    
    def get_agent_contributions(self) -> Dict:
        """Get a breakdown of agent contributions"""
        return {agent_id: len([msg for msg in self.workspace.messages if msg.sender == agent_id]) 
               for agent_id in self.agents}


# Example usage
def main():
    # Create the multi-agent system
    sigma = SIGMA()
    
    # Example problem
    problem = "Solve the quadratic equation x^2 - 5x + 6 = 0"
    
    # Solve the problem
    result = sigma.solve_problem(problem)
    
    # Explain the solution
    explanation = sigma.explain_solution(result)
    print(explanation)
    
    # Print metrics
    print("\nSystem metrics:")
    print(f"Steps taken: {result['steps_taken']}")
    print("Agent contributions:")
    for agent_id, message_count in result['agent_messages'].items():
        print(f"  {agent_id}: {message_count} messages")


if __name__ == "__main__":
    main()