from __future__ import annotations

from typing import List, Tuple
from collections import deque
from app.domain.audit import BoundingBox, DataBlockAudit, BlockCategory, BlockParsingStatus
from app.domain.document import DocumentSheet, CellDataType


class TableSegmenter:
    def __init__(self, max_adjacency_distance: int = 2):
        self.max_adj_dist = max_adjacency_distance

    def segment_sheet(self, sheet: DocumentSheet) -> List[DataBlockAudit]:
        """Discovers independent contiguous data table blocks within a worksheet using 2D BFS spatial clustering.

        Enforces structural column boundaries: BFS cannot bridge across a column that is completely empty of data.
        """
        if sheet.row_count == 0 or sheet.col_count == 0:
            return []

        # Identify completely empty columns (no text or numeric data)
        empty_cols = set()
        for c in range(sheet.col_count):
            has_data = any(
                sheet.get_cell(r, c) is not None and
                sheet.get_cell(r, c).data_type != CellDataType.EMPTY and
                bool(sheet.get_cell(r, c).formatted_value and sheet.get_cell(r, c).formatted_value.strip())
                for r in range(sheet.row_count)
            )
            if not has_data:
                empty_cols.add(c)

        # Step 1: 2D Cell Density Matrix
        visited = [[False for _ in range(sheet.col_count)] for _ in range(sheet.row_count)]
        blocks: List[DataBlockAudit] = []
        block_counter = 0

        for r in range(sheet.row_count):
            for c in range(sheet.col_count):
                cell = sheet.get_cell(r, c)
                if visited[r][c] or cell is None or cell.data_type == CellDataType.EMPTY:
                    continue

                # Step 2: Perform 2D Breadth-First Search (BFS) to find contiguous cell cluster
                cluster_cells: List[Tuple[int, int]] = []
                queue = deque([(r, c)])
                visited[r][c] = True

                while queue:
                    curr_r, curr_c = queue.popleft()
                    cluster_cells.append((curr_r, curr_c))

                    # Check adjacent cells within max_adj_dist
                    for dr in range(-self.max_adj_dist, self.max_adj_dist + 1):
                        for dc in range(-self.max_adj_dist, self.max_adj_dist + 1):
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < sheet.row_count and 0 <= nc < sheet.col_count:
                                if not visited[nr][nc]:
                                    # Enforce structural column boundary: do not bridge across fully empty columns
                                    min_c, max_c = min(curr_c, nc), max(curr_c, nc)
                                    if any(ec in empty_cols for ec in range(min_c, max_c + 1)):
                                        continue

                                    adj_cell = sheet.get_cell(nr, nc)
                                    if adj_cell and adj_cell.data_type != CellDataType.EMPTY:
                                        visited[nr][nc] = True
                                        queue.append((nr, nc))

                # Step 3: Compute Cluster Bounding Box
                r_coords = [p[0] for p in cluster_cells]
                c_coords = [p[1] for p in cluster_cells]
                bbox = BoundingBox(
                    start_row=min(r_coords),
                    end_row=max(r_coords),
                    start_col=min(c_coords),
                    end_col=max(c_coords),
                )

                # Classify structural Category based on numeric ratio
                numeric_count = sum(
                    1 for pr, pc in cluster_cells
                    if sheet.get_cell(pr, pc) and sheet.get_cell(pr, pc).is_numeric()
                )
                total_cells = len(cluster_cells)
                numeric_ratio = numeric_count / total_cells if total_cells > 0 else 0.0

                category = BlockCategory.TABLE if numeric_ratio > 0.4 else BlockCategory.TEXT
                if category == BlockCategory.TABLE and (bbox.end_row - bbox.start_row) >= 3:
                    category = BlockCategory.CURVE

                block_counter += 1
                block_id = f"block_{sheet.sheet_name}_{block_counter}"

                audit_block = DataBlockAudit(
                    block_id=block_id,
                    category=category,
                    bounding_box=bbox,
                    parsing_confidence=1.0 if numeric_ratio > 0.4 else 0.8,
                    parsing_status=BlockParsingStatus.PARSED,
                    recommended_plugin="table_segmenter_v2",
                )
                blocks.append(audit_block)

        return blocks
