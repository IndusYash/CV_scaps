import sys
from pathlib import Path
from collections import deque

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.domain.audit import BoundingBox, DataBlockAudit, BlockCategory, BlockParsingStatus
from app.domain.document import DocumentSheet, CellDataType

class StructuralTableSegmenter:
    def __init__(self, max_adjacency_distance: int = 2):
        self.max_adj_dist = max_adjacency_distance

    def segment_sheet(self, sheet: DocumentSheet):
        if sheet.row_count == 0 or sheet.col_count == 0:
            return []

        # Find columns that are completely empty of numeric data
        empty_cols = set()
        for c in range(sheet.col_count):
            has_numeric = any(
                sheet.get_cell(r, c) and sheet.get_cell(r, c).is_numeric()
                for r in range(sheet.row_count)
            )
            has_text = any(
                sheet.get_cell(r, c) and sheet.get_cell(r, c).formatted_value and sheet.get_cell(r, c).formatted_value.strip()
                for r in range(sheet.row_count)
            )
            if not has_numeric and not has_text:
                empty_cols.add(c)

        visited = [[False for _ in range(sheet.col_count)] for _ in range(sheet.row_count)]
        blocks = []
        block_counter = 0

        for r in range(sheet.row_count):
            for c in range(sheet.col_count):
                cell = sheet.get_cell(r, c)
                if visited[r][c] or cell is None or cell.data_type == CellDataType.EMPTY:
                    continue

                cluster_cells = []
                queue = deque([(r, c)])
                visited[r][c] = True

                while queue:
                    curr_r, curr_c = queue.popleft()
                    cluster_cells.append((curr_r, curr_c))

                    for dr in range(-self.max_adj_dist, self.max_adj_dist + 1):
                        for dc in range(-self.max_adj_dist, self.max_adj_dist + 1):
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < sheet.row_count and 0 <= nc < sheet.col_count:
                                if not visited[nr][nc]:
                                    # Check if crossing a fully empty column boundary
                                    min_c, max_c = min(curr_c, nc), max(curr_c, nc)
                                    crossed_empty = any(ec in empty_cols for ec in range(min_c, max_c + 1))
                                    if crossed_empty:
                                        continue

                                    adj_cell = sheet.get_cell(nr, nc)
                                    if adj_cell and adj_cell.data_type != CellDataType.EMPTY:
                                        visited[nr][nc] = True
                                        queue.append((nr, nc))

                r_coords = [p[0] for p in cluster_cells]
                c_coords = [p[1] for p in cluster_cells]
                bbox = BoundingBox(
                    start_row=min(r_coords),
                    end_row=max(r_coords),
                    start_col=min(c_coords),
                    end_col=max(c_coords),
                )

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

wb_path = backend_dir.parent / "Ca3PI3.xlsx"
if not wb_path.exists():
    wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"

with open(wb_path, "rb") as f:
    contents = f.read()

doc_model = ExcelInputReader().read(wb_path.name, contents)
segmenter = StructuralTableSegmenter()

for s in doc_model.sheets:
    blocks = segmenter.segment_sheet(s)
    print(f"Sheet: '{s.sheet_name}' -> Blocks Count ({len(blocks)}):")
    for b in blocks:
        print(f"  Block ID='{b.block_id}', BBox: rows={b.bounding_box.start_row}..{b.bounding_box.end_row}, cols={b.bounding_box.start_col}..{b.bounding_box.end_col}")
