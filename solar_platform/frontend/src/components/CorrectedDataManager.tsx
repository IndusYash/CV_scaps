import React from 'react';

export interface CorrectedWorkbook {
  workbook_name: string;
  eligible_rows: number;
  excluded_rows: number;
}

interface Props {
  workbooks: CorrectedWorkbook[];
  busy: boolean;
  onRemove: (name: string) => void;
  onClearAll: () => void;
}

const dangerButton: React.CSSProperties = {
  padding: '0.35rem 0.8rem',
  borderRadius: '8px',
  border: '1px solid #FCA5A5',
  backgroundColor: '#FEF2F2',
  color: '#DC2626',
  fontWeight: 600,
  fontSize: '0.8rem',
  cursor: 'pointer',
};

export const CorrectedDataManager: React.FC<Props> = ({ workbooks, busy, onRemove, onClearAll }) => {
  const totalUsable = workbooks.reduce((n, w) => n + w.eligible_rows, 0);

  return (
    <div className="ui-card" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem', flexWrap: 'wrap' }}>
        <div>
          <h4 style={{ margin: 0, fontSize: '1rem', color: 'var(--color-primary)' }}>
            Stored Workbooks (Corrected Pipeline)
          </h4>
          <div style={{ fontSize: '0.8rem', color: 'var(--color-secondary)', marginTop: '0.3rem', maxWidth: '640px' }}>
            Training uses exactly the workbooks listed here. Add one with Upload New Dataset (top right),
            remove one to leave it out of training.
          </div>
        </div>
        <button
          type="button"
          disabled={busy || workbooks.length === 0}
          onClick={onClearAll}
          style={{ ...dangerButton, opacity: busy || workbooks.length === 0 ? 0.5 : 1, cursor: busy || workbooks.length === 0 ? 'not-allowed' : 'pointer' }}
        >
          Clear all
        </button>
      </div>

      {workbooks.length === 0 ? (
        <div style={{ marginTop: '1rem', padding: '1rem', borderRadius: '8px', backgroundColor: 'var(--color-accent-light)', color: 'var(--color-secondary)', fontSize: '0.9rem' }}>
          No workbooks stored yet. Upload a workbook, then return to this tab.
        </div>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '1rem', fontSize: '0.9rem' }}>
          <thead>
            <tr style={{ textAlign: 'left', color: 'var(--color-secondary)', fontSize: '0.75rem', textTransform: 'uppercase' }}>
              <th style={{ padding: '0.5rem 0' }}>Workbook</th>
              <th style={{ padding: '0.5rem 0' }}>Rows used</th>
              <th style={{ padding: '0.5rem 0' }}>Rows excluded</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {workbooks.map((w) => (
              <tr key={w.workbook_name} style={{ borderTop: '1px solid var(--color-border)' }}>
                <td style={{ padding: '0.6rem 0', fontWeight: 600, color: 'var(--color-primary)' }}>{w.workbook_name}</td>
                <td style={{ padding: '0.6rem 0' }}>{w.eligible_rows}</td>
                <td style={{ padding: '0.6rem 0', color: 'var(--color-secondary)' }}>{w.excluded_rows}</td>
                <td style={{ padding: '0.6rem 0', textAlign: 'right' }}>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => onRemove(w.workbook_name)}
                    style={{ ...dangerButton, opacity: busy ? 0.5 : 1, cursor: busy ? 'not-allowed' : 'pointer' }}
                  >
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {workbooks.length > 0 && (
        <div style={{ marginTop: '0.8rem', fontSize: '0.8rem', color: 'var(--color-secondary)' }}>
          {workbooks.length} workbook(s), {totalUsable} usable rows. Excluded rows are those the data checks rejected.
        </div>
      )}
    </div>
  );
};