use std::marker::PhantomData;
use std::rc::Rc;

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect, Alignment},
    widgets::{Block, BorderType, Borders, Padding, block::Position},
    Frame, style::{Color, Style},
};

trait RegionRectInfo {
    /// Returns the index of the rectangle in the region
    fn index(&self) -> usize;

    /// Returns the title of the rectangle
    fn title(&self) -> &'static str;

    /// Returns the height percentage of the frame for the rectangle
    fn height_percentage(&self) -> u16;
}

pub(crate) enum LeftRegionPosition {
    Top,
    Middle,
    Bottom,
}

impl RegionRectInfo for LeftRegionPosition {
    fn index(&self) -> usize {
        match self {
            LeftRegionPosition::Top => 0,
            LeftRegionPosition::Middle => 1,
            LeftRegionPosition::Bottom => 2,
        }
    }

    fn title(&self) -> &'static str {
        match self {
            LeftRegionPosition::Top => "Backend",
            LeftRegionPosition::Middle => "Benches",
            LeftRegionPosition::Bottom => "Action",
        }
    }

    fn height_percentage(&self) -> u16 {
        match self {
            LeftRegionPosition::Top => 45,
            LeftRegionPosition::Middle => 45,
            LeftRegionPosition::Bottom => 10,
        }
    }
}

pub(crate) enum RightRegionPosition {
    Top,
    Bottom,
}

impl RegionRectInfo for RightRegionPosition {
    fn index(&self) -> usize {
        match self {
            RightRegionPosition::Top => 0,
            RightRegionPosition::Bottom => 1,
        }
    }

    fn title(&self) -> &'static str {
        match self {
            RightRegionPosition::Top => "Results",
            RightRegionPosition::Bottom => "Progress",
        }
    }

    fn height_percentage(&self) -> u16 {
        match self {
            RightRegionPosition::Top => 90,
            RightRegionPosition::Bottom => 10,
        }
    }
}

pub(crate) struct Region<P: RegionRectInfo> {
    rects: Rc<[Rect]>,
    _p: PhantomData<P>,
}

pub(crate) struct Regions<L: RegionRectInfo, R: RegionRectInfo> {
    pub left: Region<L>,
    pub right: Region<R>,
}

impl Regions<LeftRegionPosition, RightRegionPosition> {
    pub fn new(frame: &Frame) -> Self {
        let outer_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![
                Constraint::Percentage(Region::<LeftRegionPosition>::width_percentage()),
                Constraint::Percentage(Region::<RightRegionPosition>::width_percentage())
            ])
            .split(frame.size());
        let left_rects = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Percentage(LeftRegionPosition::Top.height_percentage()),
                Constraint::Percentage(LeftRegionPosition::Middle.height_percentage()),
                Constraint::Percentage(LeftRegionPosition::Bottom.height_percentage()),
            ])
            .split(outer_layout[0]);
        let right_rects = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![
                Constraint::Percentage(RightRegionPosition::Top.height_percentage()),
                Constraint::Percentage(RightRegionPosition::Bottom.height_percentage()),
            ])
            .split(outer_layout[1]);
        Self::new_with_rect(left_rects, right_rects)
    }

    pub fn draw(&self, frame: &mut Frame) {
        // Left region
        frame.render_widget(
            self.left.block(LeftRegionPosition::Top),
            self.left.get_rect(LeftRegionPosition::Top),
        );
        frame.render_widget(
            self.left.block(LeftRegionPosition::Middle),
            self.left.get_rect(LeftRegionPosition::Middle),
        );
        frame.render_widget(
            self.left.block(LeftRegionPosition::Bottom),
            self.left.get_rect(LeftRegionPosition::Bottom),
        );
        // Right region
        frame.render_widget(
            self.right.block(RightRegionPosition::Top),
            self.right.get_rect(RightRegionPosition::Top),
        );
        frame.render_widget(
            self.right.block(RightRegionPosition::Bottom),
            self.right.get_rect(RightRegionPosition::Bottom),
        );
    }
}

impl<L: RegionRectInfo, R: RegionRectInfo> Regions<L, R> {
    fn new_with_rect(left_rects: Rc<[Rect]>, right_rects: Rc<[Rect]>) -> Self {
        Self {
            left: Region {
                rects: left_rects,
                _p: PhantomData,
            },
            right: Region {
                rects: right_rects,
                _p: PhantomData,
            },
        }
    }
}

impl Region<LeftRegionPosition> {
    pub fn width_percentage() -> u16 {
        25
    }
}

impl Region<RightRegionPosition> {
    pub fn width_percentage() -> u16 {
        100 - Region::<LeftRegionPosition>::width_percentage()
    }
}

impl<P: RegionRectInfo> Region<P> {
    pub fn get_rect(&self, position: P) -> Rect {
        self.rects[position.index()]
    }

    /// Widget to draw the style of a region
    fn block(&self, position: P) -> Block {
        Block::default()
            .title(position.title())
            .title_position(Position::Top)
            .title_alignment(Alignment::Center)
            .borders(Borders::all())
            .border_style(Style::default().fg(Color::DarkGray))
            .border_type(BorderType::Rounded)
            .padding(Padding {
                left: 10,
                right: 10,
                top: 2,
                bottom: 2,
            })
            .style(Style::default().bg(Color::Black))
    }
}

fn create_region_block(title: &str) -> Block {
    todo!()
}
